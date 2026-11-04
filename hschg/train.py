"""Train HSCHG on cached visual/text/Mel inputs."""
import argparse,json,os,random,signal,time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from .network import load_model,portable_state
from .model import objective,clip_groups
from .backbone import load_audio,encode_audio
from .data import TrainingInputs,worker_init,sha256
from .evaluate import evaluate


def rng():return dict(python=random.getstate(),numpy=np.random.get_state(),cpu=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all())


def restore_rng(s):
    random.setstate(s['python']);np.random.set_state(s['numpy']);torch.set_rng_state(s['cpu']);torch.cuda.set_rng_state_all(s['cuda'])


def save_atomic(value,path):
    path=Path(path);pending=path.with_name(path.name+'.pending');torch.save(value,pending);pending.replace(path)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True,help='Portable model containing the frozen T01 reference')
    p.add_argument('--train-manifest',required=True);p.add_argument('--val-manifest',required=True)
    p.add_argument('--imagebind-source',required=True);p.add_argument('--imagebind-weights',required=True)
    p.add_argument('--output',required=True);p.add_argument('--device',default='cuda:0');p.add_argument('--workers',type=int,default=4)
    p.add_argument('--resume',action='store_true');a=p.parse_args()
    if not a.device.startswith('cuda') or not torch.cuda.is_available():raise ValueError('Training requires a CUDA device')
    if a.workers<0:raise ValueError('Nonnegative worker count required')
    if json.loads(Path(a.val_manifest).read_text()).get('split')!='val':
        raise ValueError('Training/model selection accepts validation only, never test')
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    torch.cuda.set_device(torch.device(a.device));torch.cuda.set_per_process_memory_fraction(.9,torch.device(a.device))
    output=Path(a.output).resolve()
    if output.is_relative_to(Path(__file__).resolve().parents[1]):raise ValueError('Write experiment files outside the source repository')
    if output.exists() and any(output.iterdir()) and not a.resume:raise ValueError('Use a new output directory or explicit --resume')
    output.mkdir(parents=True,exist_ok=True)
    model,original=load_model(a.checkpoint,a.device,fresh=True)
    data=TrainingInputs(a.train_manifest,model.head.seen_ids)
    if len(data)!=13182:raise ValueError('Released one-epoch protocol expects 13182 training videos')
    encoder=load_audio(a.imagebind_source,a.imagebind_weights,a.device)
    params=[p for p in model.parameters() if p.requires_grad]
    optimizer=torch.optim.Adam(params,lr=5e-5)
    scheduler=torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=[10,20,30],gamma=.5)
    generator=torch.Generator().manual_seed(123)
    orders=[x.tolist() for x in DataLoader(range(len(data)),batch_size=32,shuffle=True,num_workers=0,generator=generator)]
    identity=dict(version=model.config.version,config=model.config.to_dict(),seed=123,batch=32,epochs=1,
        checkpoint_sha256=sha256(a.checkpoint),train_manifest_sha256=sha256(a.train_manifest),val_manifest_sha256=sha256(a.val_manifest),
        imagebind_sha256=sha256(a.imagebind_weights),orders=orders,workers=a.workers,
        source_sha256={p.name:sha256(p) for p in Path(__file__).parent.glob('*.py')})
    random.seed(123);np.random.seed(123);torch.manual_seed(123);torch.cuda.manual_seed_all(123)
    state=dict(steps=0,phase='train');stop=[False]
    signal.signal(signal.SIGTERM,lambda *unused:stop.__setitem__(0,True));signal.signal(signal.SIGINT,lambda *unused:stop.__setitem__(0,True))
    def save():
        save_atomic(dict(identity=identity,state=state,model=model.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),rng=rng()),output/'last.pt')
    if a.resume:
        ck=torch.load(output/'last.pt',map_location='cpu')
        if ck['identity']!=identity:raise ValueError('Resume identity mismatch')
        model.load_state_dict(ck['model']);optimizer.load_state_dict(ck['optimizer']);scheduler.load_state_dict(ck['scheduler']);state=ck['state'];restore_rng(ck['rng'])
    if state['phase']=='completed':raise ValueError('This experiment already completed')
    if state['phase']=='train':
        extra=dict(persistent_workers=True,prefetch_factor=1,multiprocessing_context='spawn',worker_init_fn=worker_init) if a.workers else {}
        loader=DataLoader(data,batch_sampler=orders[state['steps']:],num_workers=a.workers,pin_memory=True,
            generator=torch.Generator().manual_seed(12300),**extra)
        model.train();ids=list(model.head.seen_ids)
        for i,(mel,vision,target,names) in enumerate(loader,state['steps']):
            if list(names)!=[data.rows[j]['video'] for j in orders[i]]:raise ValueError('Training video order mismatch')
            xa=encode_audio(encoder,mel.to(a.device,non_blocking=True));xv=vision.to(a.device,non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            out=model(xa,xv,model.head.seen_text,ids,ids.index('other'));loss,terms=objective(model.head,out,target.to(a.device,non_blocking=True))
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite objective')
            loss.backward();norms=clip_groups(model.temporal,model.head);optimizer.step();torch.cuda.synchronize()
            state['steps']=i+1
            if i==0 or (i+1)%10==0 or i==411:print(json.dumps(dict(batch=i+1,batches=412,loss=float(loss),terms={k:float(v) for k,v in terms.items()},gradients=norms)),flush=True)
            if (i+1)%25==0 or stop[0]:save()
            if stop[0]:raise SystemExit('Stopped after committed batch. Use explicit --resume.')
        del loader
        state['phase']='validation';save()
    # Frozen feature validation consumes no training RNG or online encoder calls.
    metrics,arrays=evaluate(model,a.val_manifest,a.device)
    if state['steps']!=412:raise ValueError('Incomplete training')
    scheduler.step();state.update(phase='completed',metrics=metrics);save()
    save_atomic(portable_state(model,dict(metrics=metrics,seed=123,baseline_updates=412,adaptation_updates=412,test_used=False)),output/'model.pt')
    print(json.dumps(dict(validation=metrics,steps=412,checkpoint=str(output/'model.pt')),indent=2))


if __name__=='__main__':main()
