"""One original-order cached inference and official seen/unseen/total aggregation."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from .network import load_model
from .data import validation_batches
from .metrics import all_metrics


@torch.no_grad()
def evaluate(model,manifest,device='cpu'):
    model.eval();pred=[];target=[];types=[];videos=[]
    for i,b,row,ids in validation_batches(manifest):
        a,v,text=(b[k].to(device) for k in ['audio','vision','text'])
        scores=model(a,v,text,ids,ids.index('other'))['scores']
        if not torch.isfinite(scores).all():raise FloatingPointError('Nonfinite scores')
        pred.append(scores.argmax(-1).cpu().numpy());target.append(b['labels'][0].numpy())
        types.extend(row['types']);videos.extend(row['videos'])
        if i==0 or (i+1)%10==0:print('Evaluation batch',i+1,flush=True)
    pred,target=np.concatenate(pred),np.concatenate(target);types=np.array(types)
    metrics,seg,eve=all_metrics(pred,target,types,classes=len(ids))
    return metrics,dict(pred=pred,target=target,types=types,videos=np.array(videos),seg=seg,eve=eve)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoint',required=True);p.add_argument('--manifest',required=True)
    p.add_argument('--device',default='cpu');p.add_argument('--predictions',help='Optional NPZ destination outside the source repository')
    a=p.parse_args();torch.set_num_threads(2);model,_=load_model(a.checkpoint,a.device)
    metrics,arrays=evaluate(model,a.manifest,a.device);print(json.dumps(metrics,indent=2))
    if a.predictions:
        path=Path(a.predictions)
        if path.exists():raise ValueError('Preserve existing predictions')
        path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(path,**arrays)


if __name__=='__main__':main()
