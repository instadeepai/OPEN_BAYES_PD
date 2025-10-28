import torch
from omegaconf import DictConfig
from torch.nn import functional

from src.models.core.mlp import MLP


class EnrichmentVariancePredictor(torch.nn.Module):
    """
    Outputs a bounded (positive) value given initial and selected frequencies of a
    sequence. This value will later be used as an estimate of the variance around the
    experimentally estimated enrichment ratio of a serquence.
    """

    def __init__(self, cfg: DictConfig):
        """Initialize"""
        super().__init__()
        self.enrichment_variance_predictor = MLP(**cfg.model.enrichment_variance)

    def forward(self, variance_predictor_input: torch.Tensor):
        """Forward"""
        unbounded_enrichment_variance = self.enrichment_variance_predictor(
            variance_predictor_input
        )
        return functional.sigmoid(unbounded_enrichment_variance).to(torch.float64)
