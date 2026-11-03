"""Unchanged official cosine-product, ReLU, square-root, min-max head."""
from contextlib import contextmanager
import torch
import torch.nn.functional as F

@contextmanager
def isolated_cpu_seed(seed):
    with torch.random.fork_rng(devices=[]):
        torch.set_rng_state(torch.Generator(device='cpu').manual_seed(seed).get_state())
        yield

def cosine_scores(features, text):
    (b, t, d) = features.shape
    return F.cosine_similarity(features.unsqueeze(2).expand(b, t, len(text), d), text[None, None].expand(b, t, -1, -1), dim=-1)

def head_components(audio, vision, text):
    q = torch.sqrt(F.relu(cosine_scores(audio, text) * cosine_scores(vision, text)))
    (lo, hi) = (q.min(-1, keepdim=True).values, q.max(-1, keepdim=True).values)
    return ((q - lo) / (hi - lo + 1e-08), q - lo, hi - lo)

def official_head(audio, vision, text):
    return head_components(audio, vision, text)[0]
