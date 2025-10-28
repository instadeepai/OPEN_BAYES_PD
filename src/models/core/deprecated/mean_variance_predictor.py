import torch
from omegaconf import DictConfig
from torch.nn import functional

from src.models.core.mlp import MLP
from src.models.core.resnext import ResNeXtModel


def _get_fitness_predictor(cfg: DictConfig) -> tuple[torch.nn.Module, int]:
    """
    Select chosen architecture for the fitness predictor used in flighted and
    instantiate.
    """
    if cfg.model.fitness_predictor_architecture == "MLP":
        return (
            MLP(**cfg.model.fitness_predictor_mlp),
            cfg.model.fitness_predictor_mlp.layers_size[-1],
        )
    if cfg.model.fitness_predictor_architecture == "ResNext":
        return (
            ResNeXtModel(**cfg.model.fitness_predictor_resnext),
            cfg.model.fitness_predictor_resnext.resnext_mlp_layers_size[-1],
        )
    raise NotImplementedError


class FitnessMeanVariancePredictor(torch.nn.Module):
    """Map a sequence to its predicted average
    fitness (for now : binding probability) and the predicted uncertainty around that
    average.
    """

    def __init__(self, cfg: DictConfig):
        """Initialize"""
        super().__init__()
        self._fitness_predictor, last_layer_size = _get_fitness_predictor(cfg)
        self._linear_mean = torch.nn.Linear(last_layer_size, 1)
        self._linear_variance = torch.nn.Linear(last_layer_size + 1, 1)

    def forward(self, sequence_representation: torch.Tensor):
        """Forward"""
        fitness_predicted = self._fitness_predictor(sequence_representation)
        fitness_mean = self._linear_mean(fitness_predicted)
        fitness_variance = functional.sigmoid(
            self._linear_variance(torch.hstack([fitness_predicted, fitness_mean]))
        )
        return fitness_mean, fitness_variance
