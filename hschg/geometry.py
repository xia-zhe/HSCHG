"""Lorentz exponential and entailment cost used by V999-repair-v1.

Scalar geometry calculations use FP64; model parameters remain FP32.
Role-specific radii and the bounded shared map are defined in model.py.
"""
import math
import torch


def exp_map0(q, curvature):
    q = q.double()
    r2 = curvature*q.square().sum(-1, keepdim=True)
    safe = r2.clamp_min(1e-12).sqrt()
    small = r2 <= 1e-8
    scale = torch.where(small, 1+r2/6+r2.square()/120, safe.sinh()/safe)
    time = torch.where(small, 1+r2/2+r2.square()/24, safe.cosh())/math.sqrt(curvature)
    result = torch.cat((time, scale*q), -1)
    if not torch.isfinite(result).all():
        raise FloatingPointError('V999 Lorentz exponential is nonfinite; no silent radius clamp')
    return result


def entailment_cost(parent, child, curvature, kappa, epsilon):
    parent, child = parent.double(), child.double()
    xp, yp = parent[..., 1:], child[..., 1:]
    x0, y0 = parent[..., 0], child[..., 0]
    # The zero-norm branch has the exact value zero and finite zero derivative.
    norm2 = xp.square().sum(-1)
    norm = torch.where(norm2 == 0, torch.zeros_like(norm2), norm2.clamp_min(1e-30).sqrt())
    b = curvature*(x0*y0-(xp*yp).sum(-1))
    angular = (y0-b*x0)/(norm*(b.square()-1).clamp_min(epsilon).sqrt()+epsilon)
    theta = angular.clamp(-1+epsilon, 1-epsilon).acos()
    aperture_arg = 2*kappa/(math.sqrt(curvature)*norm+epsilon)
    aperture = aperture_arg.clamp(-1+epsilon, 1-epsilon).asin()
    cost = (theta-aperture).clamp_min(0)
    if not torch.isfinite(cost).all():
        raise FloatingPointError('Nonfinite PDF entailment cost')
    return cost
