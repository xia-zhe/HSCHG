"""PDF Eqs.3-10: relation-local normalization and synchronous AV updates."""
import torch
from torch import nn
import torch.nn.functional as F


def masked_softmax(logits, mask):
    """An empty selected set yields an exactly zero message, not NaN."""
    masked = logits.masked_fill(~mask, -torch.inf)
    valid = mask.any(-1, keepdim=True)
    safe = torch.where(valid, masked, torch.zeros_like(masked))
    return safe.softmax(-1)*mask.to(logits.dtype)


def similarities(target, source):
    return F.normalize(target, dim=-1, eps=1e-8) @ F.normalize(source, dim=-1, eps=1e-8).transpose(-1, -2)


def temporal_message(value, radius, threshold):
    steps = value.shape[1]
    time = torch.arange(steps, device=value.device)
    offset = time[:, None]-time[None]
    similarity = similarities(value, value)
    neighborhoods = ((offset.abs() > 0) & (offset.abs() <= radius),
                     (offset > 0) & (offset <= radius),
                     (offset < 0) & (-offset <= radius))
    messages = []
    for neighborhood in neighborhoods:
        selected = neighborhood[None] & (similarity >= threshold)
        messages.append(masked_softmax(similarity, selected) @ value)
    return sum(messages)/3


def cross_message(target, source, config):
    steps = target.shape[1]
    time = torch.arange(steps, device=target.device)
    neighbors = ((time[:, None]-time[None]).abs() <= config.cross_radius)[None]
    similarity = similarities(target, source)
    # Eq.6 normalizes over ALL candidate neighbors, before threshold weighting.
    attention = masked_softmax(similarity, neighbors.expand_as(similarity))
    graded = torch.zeros_like(similarity)
    graded = torch.where(similarity >= config.cross_threshold_low, config.cross_weight_low, graded)
    graded = torch.where(similarity >= config.cross_threshold_high, config.cross_weight_high, graded)
    return (attention*graded) @ source


class GraphLayer(nn.Module):
    def __init__(self, config, cross_scale):
        super().__init__()
        self.config, self.cross_scale = config, cross_scale
        d = config.feature_dim
        self.ffn_audio = nn.Sequential(nn.Linear(d, d, bias=False), nn.GELU(), nn.Linear(d, d, bias=False))
        self.ffn_vision = nn.Sequential(nn.Linear(d, d, bias=False), nn.GELU(), nn.Linear(d, d, bias=False))
        self.gate_audio = nn.Linear(2*d, d, bias=False)
        self.gate_vision = nn.Linear(2*d, d, bias=False)
        self.context_audio = nn.Linear(d, d, bias=False)
        self.context_vision = nn.Linear(d, d, bias=False)
        self.pool_audio = nn.Linear(d, 1, bias=False)
        self.pool_vision = nn.Linear(d, 1, bias=False)
        self.project_audio = nn.Linear(d, d, bias=False)
        self.project_vision = nn.Linear(d, d, bias=False)
        # PDF does not specify LN sharing. Use the old source's per-modality
        # stage sharing; independent layers own all their own parameters.
        self.norm_audio = nn.LayerNorm(d)
        self.norm_vision = nn.LayerNorm(d)
        self.norm_video_audio = nn.LayerNorm(d)
        self.norm_video_vision = nn.LayerNorm(d)

    def forward(self, audio, vision, video_audio, video_vision):
        c = self.config
        a = self.norm_audio(audio+self.ffn_audio(temporal_message(audio, c.temporal_radius, c.intra_threshold)))
        v = self.norm_vision(vision+self.ffn_vision(temporal_message(vision, c.temporal_radius, c.intra_threshold)))
        # Both incoming messages use the SAME pre-cross states a and v.
        ma, mv = cross_message(a, v, c), cross_message(v, a, c)
        ua = self.norm_audio(a+self.cross_scale*torch.sigmoid(self.gate_audio(torch.cat((a, ma), -1)))*ma)
        uv = self.norm_vision(v+self.cross_scale*torch.sigmoid(self.gate_vision(torch.cat((v, mv), -1)))*mv)
        # Eq.8 deliberately reuses W_c^a three times, including the visual
        # cross-modal term. Do not silently symmetrize to the older source.
        ca = self.context_audio(video_audio)+self.context_audio(video_vision)
        cv = self.context_vision(video_vision)+self.context_audio(video_audio)
        next_audio = self.norm_audio(ua+c.context_scale*ca[:, None])
        next_vision = self.norm_vision(uv+c.context_scale*cv[:, None])
        pa = self.pool_audio(next_audio).softmax(1)
        pv = self.pool_vision(next_vision).softmax(1)
        next_video_audio = self.norm_video_audio(video_audio+(pa*self.project_audio(next_audio)).sum(1))
        next_video_vision = self.norm_video_vision(video_vision+(pv*self.project_vision(next_vision)).sum(1))
        return next_audio, next_vision, next_video_audio, next_video_vision


class HeterogeneousGraph(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.ModuleList([GraphLayer(config, scale) for scale in config.cross_scales])

    def forward(self, audio, vision):
        video_audio, video_vision = audio.mean(1), vision.mean(1)
        for layer in self.layers:
            audio, vision, video_audio, video_vision = layer(audio, vision, video_audio, video_vision)
        return audio, vision, video_audio, video_vision
