"""Frozen online audio adapter to the external, pinned OV-AVEL release."""
import hashlib,sys
from pathlib import Path
import torch

PINNED_FILES={
 'imagebind_finetune/models/imagebind_model_fully.py':'391668771ee38e7165e187ecf187f5397f95b46b3adec09ec4ef2edcfc238704',
 'imagebind_finetune/models/transformer.py':'1e721df23995f20f802b79bd3fad85a9b7e6e83a7b2c202bbda91971b78fcad7',
 'layers/temporal_av_attn_layer.py':'0d587b054b8ad0e2bf9484f8a84863abbfe55a1ca474e81c9f9e7fdc3fd2948a'}


def load_audio(source,weights,device):
    """source is OV-AVEL/proposed_method/ImageBind-main, kept outside this repo."""
    source=Path(source).resolve()
    from .data import sha256
    if sha256(weights)!='d6f6c22bedcc90708448d5d2fbb7b2db9c73f505dc89bd0b2e09b23af1b62157':
        raise ValueError('Use the frozen ImageBind huge weights matching this release')
    for name,expected in PINNED_FILES.items():
        if hashlib.sha256((source/name).read_bytes()).hexdigest()!=expected:
            raise ValueError('Use pinned OV-AVEL source: '+name)
    sys.path.insert(0,str(source))
    from imagebind_finetune.models.imagebind_model_fully import imagebind_huge
    model=imagebind_huge(pretrained=False,spatial_av_attn_layer_ids=([],[]),sattn_flag='none',
        tattn_flag=True,sa_layer_num=1,xa_layer_num=1,hid_dim=256,d_ff=512,head_num=1,
        dropout=.1,use_adj_in_attn=False,gamma=.6,bias=.2,use_mask_in_attn=False,
        win_size=5,norm_flag=None,text_tune_flag=False)
    state=torch.load(weights,map_location='cpu');own=model.state_dict()
    selected={k:v for k,v in state.items() if k in own}
    missing,unexpected=model.load_state_dict(selected,strict=False)
    if unexpected or any(not k.startswith(('temporal_av_layer.','task_res')) for k in missing):
        raise ValueError('ImageBind pretrained state mismatch')
    del state,selected,own
    blocks=model.modality_trunks['audio'].blocks
    def wrap(original):
        def forward(*a,**kw):
            with torch.backends.cuda.sdp_kernel(enable_flash=False,enable_math=True,enable_mem_efficient=False):
                return original(*a,**kw)
        return forward
    for block in blocks:
        if not isinstance(block.attn,torch.nn.MultiheadAttention):raise ValueError('Unexpected audio attention')
        block.attn.forward=wrap(block.attn.forward)
    model.requires_grad_(False).train()
    for group in ['modality_preprocessors','modality_trunks','modality_heads','modality_postprocessors']:
        getattr(model,group)['audio'].to(device)
    return model


@torch.no_grad()
def encode_audio(model,value):
    if value.ndim!=5:raise ValueError('Expected [B,10,1,128,204] cached Mel input')
    b,t=value.shape[:2];value=value.reshape(b*t,*value.shape[2:])
    prepared=model.modality_preprocessors['audio'](audio=value)
    tokens=prepared['trunk']['tokens'];trunk=model.modality_trunks['audio']
    if trunk.pre_transformer_layer:tokens=trunk.pre_transformer_layer(tokens)
    for block in trunk.blocks:tokens=block(tokens,attn_mask=None)
    if trunk.post_transformer_layer:tokens=trunk.post_transformer_layer(tokens)
    value=model.modality_heads['audio'](tokens,**prepared['head'])
    return model.modality_postprocessors['audio'](value).reshape(b,t,-1)
