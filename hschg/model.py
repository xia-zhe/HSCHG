"""Baseline-preserving graph adaptation and scale-calibrated entailment."""
from dataclasses import dataclass, asdict
import math
import torch
from torch import nn
import torch.nn.functional as F
from .heads import official_head, isolated_cpu_seed
from .graph_config import ModelConfig as PaperConfig
from .graph import HeterogeneousGraph
from .geometry import exp_map0, entailment_cost


@dataclass(frozen=True)
class Config:
    version: str = 'V999-repair-v1'
    feature_dim: int = 1024
    graph_enabled: bool = True
    temporal_bound: float = .1
    graph_bound: float = .1
    map_bound: float = .1
    text_radius: float = .5
    video_radius: float = 1.
    segment_radius: float = 1.5
    curvature: float = 1.
    aperture_kappa: float = .1
    epsilon: float = 1e-8
    entailment_weight: float = .1
    loss_temperature: float = .1
    initialization_seed: int = 999123

    def __post_init__(self):
        if not 0 < self.map_bound < 1:
            raise ValueError('Semantic map must remain invertible')
        if not 0 < self.text_radius < self.video_radius < self.segment_radius:
            raise ValueError('Require text < video < segment radii')
        if self.curvature <= 0 or self.aperture_kappa <= 0 or not 0 < self.epsilon < 1:
            raise ValueError('Invalid geometry constants')
        if math.sinh(math.sqrt(self.curvature)*self.text_radius) <= 2*self.aperture_kappa:
            raise ValueError('Smallest radius lies outside unclipped aperture domain')
        if min(self.temporal_bound,self.graph_bound,self.entailment_weight) < 0 or self.loss_temperature <= 0:
            raise ValueError('Invalid loss/update constant')
        if not self.graph_enabled and self.entailment_weight:
            raise ValueError('Disabled path disables auxiliary loss')

    def to_dict(self):
        return asdict(self)


class Geometry(nn.Module):
    def __init__(self, config):
        super().__init__();self.config=config
        self.delta=nn.Parameter(torch.zeros(config.feature_dim,config.feature_dim))
        self.register_buffer('identity',torch.eye(config.feature_dim))

    def matrix(self):
        # ||W-I||_2 <= ||W-I||_F < map_bound; no low-rank collapse.
        return self.identity+self.config.map_bound*self.delta/(1+self.delta.square().sum()).sqrt()

    def forward(self, value, role):
        radius=getattr(self.config,role+'_radius')
        # Ignore ImageBind's modality/logit scaling only in auxiliary geometry.
        direction=F.normalize(F.linear(F.normalize(value,dim=-1,eps=1e-8),self.matrix()),dim=-1,eps=1e-8)
        return exp_map0(radius*direction,self.config.curvature)

    def cost(self,parent,child):
        c=self.config
        return entailment_cost(parent,child,c.curvature,c.aperture_kappa,c.epsilon)


class Repair(nn.Module):
    def __init__(self,config,seen_text,seen_ids,video_text=None):
        super().__init__();self.config=config;self.seen_ids=tuple(seen_ids)
        if len(set(seen_ids))!=len(seen_ids) or self.seen_ids.count('other')!=1:
            raise ValueError('Unique class identities and one other are required')
        if seen_text.shape!=(len(seen_ids),config.feature_dim):raise ValueError('Text shape')
        self.foreground_ids=tuple(x for x in self.seen_ids if x!='other')
        foreground=[i for i,x in enumerate(self.seen_ids) if x!='other']
        self.register_buffer('seen_text',seen_text.detach().cpu().clone())
        # The segment parent is exactly the classifier's original class text.
        self.register_buffer('segment_text',seen_text[foreground].detach().cpu().clone())
        if config.graph_enabled:
            if video_text is None or video_text.shape!=(len(foreground),config.feature_dim):raise ValueError('Video text shape')
            self.register_buffer('video_text',video_text.detach().cpu().clone())
            with isolated_cpu_seed(config.initialization_seed):
                self.graph=HeterogeneousGraph(PaperConfig(feature_dim=config.feature_dim))
                self.output_audio=nn.Linear(config.feature_dim,config.feature_dim,bias=False)
                self.output_vision=nn.Linear(config.feature_dim,config.feature_dim,bias=False)
                nn.init.zeros_(self.output_audio.weight);nn.init.zeros_(self.output_vision.weight)
                self.geometry=Geometry(config)

    def _correct(self,reference,temporal,graph,projection):
        scale=reference.detach().norm(dim=-1,keepdim=True)/math.sqrt(self.config.feature_dim)
        dt=self.config.temporal_bound*scale*torch.tanh(temporal-reference.detach())
        dg=self.config.graph_bound*scale*torch.tanh(projection(F.layer_norm(graph,(self.config.feature_dim,))))
        return reference.detach()+dt+dg

    def forward(self,xa,xv,ha,hv,text,candidate_ids,bg_index,*,reference_audio=None,reference_vision=None):
        ids=tuple(candidate_ids)
        if len(ids)!=len(text) or len(set(ids))!=len(ids) or ids[bg_index]!='other':raise ValueError('Candidate identity')
        if self.training:
            if set(ids)!=set(self.seen_ids):raise RuntimeError('Unseen text cannot enter training')
            order=[self.seen_ids.index(x) for x in ids]
            if not torch.equal(text,self.seen_text[order]):raise RuntimeError('Classification text changed')
        if not self.config.graph_enabled:
            return dict(scores=official_head(ha,hv,text),audio=ha,vision=hv,video_audio=ha.mean(1),video_vision=hv.mean(1),candidate_ids=ids,bg_index=bg_index)
        if reference_audio is None or reference_vision is None:raise ValueError('Frozen T01 temporal reference required')
        ga,gv,va,vv=self.graph(ha,hv)
        a=self._correct(reference_audio,ha,ga,self.output_audio)
        v=self._correct(reference_vision,hv,gv,self.output_vision)
        return dict(scores=official_head(a,v,text),audio=a,vision=v,video_audio=va,video_vision=vv,candidate_ids=ids,bg_index=bg_index)

    def entailment(self,output,target):
        if not self.training:raise RuntimeError('No labels at inference')
        zero=output['scores'].sum()*0
        if not self.config.graph_enabled or not self.config.entailment_weight:return dict(intra=zero,cross=zero,total=zero)
        bg=output['bg_index'];mask=target!=bg;counts=mask.sum(-1);active=counts>0
        if not active.any():
            zero=zero+self.geometry.delta.sum()*0
            return dict(intra=zero,cross=zero,total=zero)
        event=target.masked_fill(~mask,-1).max(-1).values
        if ((target!=bg)&(target!=event[:,None])).any():raise ValueError('Multiple foreground classes in one training video')
        names=output['candidate_ids'];lookup={name:i for i,name in enumerate(self.foreground_ids)}
        rows=torch.tensor([lookup[names[int(i)]] for i in event[active]],device=target.device)
        ep=self.geometry(self.segment_text[rows],'text');ev=self.geometry(self.video_text[rows],'text')
        intra=ep.new_zeros(len(rows));cross=ep.new_zeros(len(rows))
        for segment,video in [('audio','video_audio'),('vision','video_vision')]:
            zp=self.geometry(output[segment][active],'segment');zv=self.geometry(output[video][active],'video')
            intra+=(self.geometry.cost(zv[:,None],zp)*mask[active]).sum(-1)/counts[active]
            cross+=self.geometry.cost(ev,zv)+(self.geometry.cost(ep[:,None],zp)*mask[active]).sum(-1)/counts[active]
        intra=intra.sum()/len(target);cross=cross.sum()/len(target)
        return dict(intra=intra,cross=cross,total=intra+cross)


def objective(head,output,target):
    ce=F.cross_entropy((output['scores']/head.config.loss_temperature).permute(0,2,1),target)
    relation=head.entailment(output,target)
    return ce+head.config.entailment_weight*relation['total'],dict(classification=ce,**relation)


def clip_groups(temporal,head,maximum=.8):
    geometry=list(head.geometry.parameters()) if head.config.graph_enabled else []
    geo_ids={id(p) for p in geometry}
    main=[p for p in list(temporal.parameters())+list(head.parameters()) if p.requires_grad and id(p) not in geo_ids]
    return dict(main=float(nn.utils.clip_grad_norm_(main,maximum,error_if_nonfinite=True)),
                geometry=float(nn.utils.clip_grad_norm_(geometry,maximum,error_if_nonfinite=True)) if geometry else 0.)
