"""Predict one trusted frozen-feature tensor bundle on CPU or CUDA."""
import argparse,json
import torch
from .network import load_model


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoint',required=True);p.add_argument('--features',required=True)
    p.add_argument('--device',default='cpu');a=p.parse_args();torch.set_num_threads(2)
    model,_=load_model(a.checkpoint,a.device);b=torch.load(a.features,map_location='cpu')
    ids=b['candidate_ids'];bg=ids.index('other')
    with torch.no_grad():
        scores=model(*(b[k].to(a.device) for k in ['audio','vision','text']),ids,bg)['scores']
    pred=scores.argmax(-1).cpu().tolist()
    print(json.dumps(dict(indices=pred,labels=[[ids[i] for i in row] for row in pred]),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
