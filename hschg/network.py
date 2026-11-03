"""Self-contained temporal reference, temporal adapter and graph model."""
import copy
from pathlib import Path
import torch
from torch import nn
from .model import Config,Repair
from .temporal import TemporalAttentionModule


def make_temporal():
    return TemporalAttentionModule(sa_layer_num=1,xa_layer_num=0,feat_dim=1024,
        hid_dim=256,d_ff=512,head_num=1,dropout=.1,use_adj_in_attn=False,
        gamma=.6,bias=.2,use_mask_in_attn=False,win_size=5,norm_flag=None)


class HSCHG(nn.Module):
    """Input: frozen [B,T,1024] AV features and candidate [C,1024] text."""
    def __init__(self,config,seen_text,seen_ids,video_text):
        super().__init__()
        self.config=config
        self.reference=make_temporal().eval().requires_grad_(False)
        self.temporal=make_temporal()
        self.head=Repair(config,seen_text,seen_ids,video_text)

    def train(self,mode=True):
        super().train(mode)
        self.reference.eval().requires_grad_(False)
        return self

    def forward(self,audio,vision,text,candidate_ids,bg_index):
        if audio.ndim!=3 or audio.shape!=vision.shape or audio.shape[-1]!=self.config.feature_dim:
            raise ValueError('Expected equally shaped [B,T,1024] audio/vision features')
        with torch.no_grad():
            ra,rv=self.reference(audio,vision)
        ha,hv=self.temporal(audio,vision)
        return self.head(audio,vision,ha,hv,text,candidate_ids,bg_index,
                         reference_audio=ra,reference_vision=rv)


def load_model(path,device='cpu',fresh=False):
    """Load trusted portable weights. fresh=True restarts repair from frozen B0."""
    ck=torch.load(Path(path),map_location='cpu')
    if ck.get('format')!='hschg-v999-repair-v1':
        raise ValueError('Expected the portable V999-repair-v1 checkpoint')
    config=Config(**ck['config']);ids=ck['seen_ids'];state=ck['model']
    model=HSCHG(config,state['head.seen_text'],ids,state['head.video_text'])
    if fresh:
        reference={k[len('reference.'):]:v for k,v in state.items() if k.startswith('reference.')}
        model.reference.load_state_dict(reference,strict=True)
        model.temporal.load_state_dict(reference,strict=True)
    else:
        model.load_state_dict(state,strict=True)
    model.to(device).eval()
    return model,ck


def portable_state(model,metadata=None):
    return dict(format='hschg-v999-repair-v1',config=model.config.to_dict(),
        seen_ids=list(model.head.seen_ids),
        model={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
        metadata=metadata or {})
