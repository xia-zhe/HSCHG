"""Explicit values and provenance; no unreported decoder or research modules."""
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ModelConfig:
    version: str = 'V999'
    feature_dim: int = 1024
    graph_enabled: bool = True
    graph_layers: int = 4
    temporal_radius: int = 2
    cross_radius: int = 0
    intra_threshold: float = .5
    cross_threshold_low: float = .2
    cross_threshold_high: float = .5
    cross_weight_low: float = .5
    cross_weight_high: float = .8
    cross_scales: tuple = (.15, .35, .60, 1.)
    context_scale: float = .1
    curvature: float = 1.
    aperture_kappa: float = .1
    epsilon: float = 1e-8
    entailment_weight: float = .1
    loss_temperature: float = .1
    initialization_seed: int = 999123

    def __post_init__(self):
        if self.version != 'V999' or self.feature_dim < 1:
            raise ValueError('Invalid version/dimension')
        if self.graph_layers < 1 or len(self.cross_scales) != self.graph_layers:
            raise ValueError('One explicit fixed scale per independent graph layer is required')
        if min(self.temporal_radius, self.cross_radius) < 0:
            raise ValueError('Nonnegative temporal/cross radii required')
        if not self.cross_threshold_low < self.cross_threshold_high:
            raise ValueError('Ordered cross-modal thresholds required')
        if not 0 < self.cross_weight_low < self.cross_weight_high <= 1:
            raise ValueError('Ordered fixed cross-modal weights required')
        if min(self.curvature, self.aperture_kappa, self.epsilon, self.loss_temperature) <= 0:
            raise ValueError('Positive geometric and loss constants required')
        if self.epsilon >= 1 or self.entailment_weight < 0:
            raise ValueError('Invalid epsilon/loss weight')
        if not self.graph_enabled and self.entailment_weight:
            raise ValueError('The exact baseline control disables both graph and entailment')

    def to_dict(self):
        return asdict(self)
