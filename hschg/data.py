"""Portable feature/Mel manifests; no experiment paths or hardware-bound cache keys."""
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()


def checked_path(record,key,root,checksum):
    path=Path(record[key]);path=path if path.is_absolute() else root/path
    if record.get(checksum) and sha256(path)!=record[checksum]:raise RuntimeError('Corrupt cached input: '+str(path))
    return path


class TrainingInputs(Dataset):
    def __init__(self,manifest,seen_ids):
        self.path=Path(manifest).resolve();self.manifest=json.loads(self.path.read_text())
        if self.manifest.get('format')!='hschg-training-v1' or self.manifest.get('split')!='train':raise ValueError('Training manifest format/split')
        if self.manifest['candidate_ids']!=list(seen_ids):raise ValueError('Seen training vocabulary/order differs')
        self.rows=self.manifest['entries'];self.root=self.path.parent
    def __len__(self):return len(self.rows)
    def __getitem__(self,i):
        row=self.rows[i]
        mp=checked_path(row,'mel',self.root,'mel_sha256');vp=checked_path(row,'vision',self.root,'vision_sha256')
        mel=np.load(mp,allow_pickle=False);vision=torch.load(vp,map_location='cpu')
        if mel.shape==(1,10,1,128,204):mel=mel[0]
        if mel.shape!=(10,1,128,204) or mel.dtype!=np.float32 or vision.shape!=(10,1024) or vision.dtype!=torch.float32:
            raise ValueError('Expected FP32 Mel[10,1,128,204], vision[10,1024]')
        target=torch.tensor(row['target'],dtype=torch.long)
        if target.shape!=(10,) or (target<0).any() or (target>=len(self.manifest['candidate_ids'])).any():raise ValueError('Invalid labels')
        if not np.isfinite(mel).all() or not torch.isfinite(vision).all():raise ValueError('Nonfinite cache')
        return torch.from_numpy(mel),vision,target,row['video']


def validation_batches(manifest):
    path=Path(manifest).resolve();m=json.loads(path.read_text())
    if m.get('format')!='hschg-evaluation-v1' or m['split'] not in ('val','test'):raise ValueError('Evaluation manifest')
    for i,row in enumerate(m['batches']):
        source=checked_path(row,'path',path.parent,'sha256');b=torch.load(source,map_location='cpu')
        n=len(row['videos'])
        for k,shape in [('audio',(n,10,1024)),('vision',(n,10,1024)),('text',(len(m['candidate_ids']),1024))]:
            if b[k].shape!=shape or b[k].dtype!=torch.float32 or not torch.isfinite(b[k]).all():raise ValueError('Invalid '+k+' features')
        if b['videos']!=row['videos']:raise ValueError('Video ordering differs')
        if len(row['types'])!=n:raise ValueError('Subset annotation length differs')
        yield i,b,row,m['candidate_ids']


def worker_init(unused):torch.set_num_threads(1)


def convert_cache(project,output,training_index,split='val'):
    """Read existing immutable OV-AVEL caches; write only lightweight manifests."""
    project=Path(project).resolve();output=Path(output).resolve();output.mkdir(parents=True,exist_ok=True)
    if any(output.iterdir()):raise ValueError('Use a new empty manifest output directory')
    index=json.loads(Path(training_index).read_text())
    entries=[dict(video=e['row']['vid_name'],mel=str(project/e['mel']),mel_sha256=e['mel_sha256'],
          vision=str(project/e['visual']),vision_sha256=e['visual_sha256'],target=e['target']) for e in index['entries']]
    train=dict(format='hschg-training-v1',split='train',candidate_ids=index['identity']['candidate_ids'],entries=entries)
    (output/'train.json').write_text(json.dumps(train,indent=2)+'\n')
    matching=[]
    for p in (project/'cache/research_evaluation_v1').glob('*/identity.json'):
        ident=json.loads(p.read_text())
        if ident.get('split')==split and ident.get('batch')==32:matching.append((p.parent,ident))
    if len(matching)!=1:raise ValueError('Expected exactly one original batch32 cache for '+split)
    directory,ident=matching[0]
    rows=[r for r in csv.DictReader(open(project/'official/meta_anno_files/ovave_dataset_meta.csv')) if r['split']==split]
    batches=[]
    for i in range((len(rows)+31)//32):
        p=directory/f'batch_{i:04d}.pt';marker=json.loads(p.with_suffix('.json').read_text());selected=rows[i*32:(i+1)*32]
        names=[r['vid_name'] for r in selected]
        if marker['videos']!=names:raise ValueError('Source cache video order differs')
        batches.append(dict(path=str(p),sha256=marker['sha256'],videos=names,types=[r['cls_type'] for r in selected]))
    (output/(split+'.json')).write_text(json.dumps(dict(format='hschg-evaluation-v1',split=split,candidate_ids=ident['candidates'],batches=batches),indent=2)+'\n')
    print('Created train.json and '+split+'.json; original caches unchanged')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--project',required=True);p.add_argument('--output',required=True)
    p.add_argument('--training-index',required=True,help='Path to the existing training cache index')
    p.add_argument('--split',choices=['val','test'],default='val');a=p.parse_args();convert_cache(a.project,a.output,a.training_index,a.split)
