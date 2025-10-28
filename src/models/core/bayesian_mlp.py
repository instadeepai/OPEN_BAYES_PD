"""
Shape Suffixes Documentation:

B: batch size
F: features (hidden dimensions in MLP layers)
O: output (final output dimension)

Example:
- input_features_BF: input features with batch and feature dimensions
"""

import numpy as np
import pyro.distributions as dist
import torch
from pyro.nn import PyroModule, PyroSample
from torch import nn

from src.models.core.model_config import (
    ActivationClass,
    BatchNormalizationClass,
    MLPConfig,
)

# TODO make a better gloroth initialization


def get_activation(activation_name: str) -> torch.nn.functional:
    """Retrieves the wanted activation function
    Note : need a dedicated not implemented error
    """
    if activation_name in ActivationClass.__dict__:
        return ActivationClass.__dict__[activation_name]
    raise NotImplementedError


def get_batch_norm(batch_norm_name: str) -> nn.Module:
    """
    Get the batch normalization function.
    """
    if batch_norm_name in BatchNormalizationClass.__dict__:
        return BatchNormalizationClass.__dict__[batch_norm_name]
    raise NotImplementedError


class BayesianMLP(PyroModule):
    """
    MLP consisting in linear layers wrapped in PyroModule allowing Bayesian training of
    the weights.

    Parameters:
    ----------
    layers_size (list[int]):
        size of input, hidden and output layers.
    prior_scale (float):
        variance (uncertainty) of the prior distribution over the network's weights and
        biases.
    activation_name (str):
        name of the activation function.
    device (Optional[torch.device]):
        define the device, cuda or cpu, to work with.
    """

    def __init__(
        self,
        cfg: MLPConfig,
    ):
        """
        Constructor for the MLP.
        """
        super().__init__()
        self._activation = get_activation(cfg.activation_name)
        self._dropout = nn.Dropout(cfg.dropout_value)
        assert np.all(cfg.layers_size) > 0

        # treat each layer weights as codependent
        layer_list = [
            PyroModule[nn.Linear](cfg.layers_size[idx - 1], cfg.layers_size[idx])
            for idx in range(1, len(cfg.layers_size))
        ]
        # treat weights of entire MLP as codependent
        self.layers = PyroModule[torch.nn.ModuleList](layer_list)

        mean = 0.0
        if cfg.device is not None:  # TODO: check if unnecessary
            mean = torch.tensor(0.0, device=cfg.device)

        for layer_idx, layer in enumerate(self.layers):
            layer.weight = PyroSample(
                dist.Normal(
                    mean, cfg.prior_scale * np.sqrt(2 / cfg.layers_size[layer_idx])
                )
                .expand([cfg.layers_size[layer_idx + 1], cfg.layers_size[layer_idx]])
                .to_event(2)
            )
            layer.bias = PyroSample(
                dist.Normal(mean, cfg.prior_scale)
                .expand([cfg.layers_size[layer_idx + 1]])
                .to_event(1)
            )

    def forward(self, input_features_BF: torch.Tensor) -> torch.Tensor:
        """
        Forward function: run BayesianMLP over the input data.

        Parameters:
        ----------
        input_features_BF (torch.Tensor):
            input data as features with shape (batch, features).

        Returns:
        -------
        torch.Tensor:
            binding affinity with shape (batch, output).
        """
        hidden_features_BF = self._activation(self.layers[0](input_features_BF))
        for layer in self.layers[1:-1]:
            hidden_features_BF = self._activation(layer(hidden_features_BF))
            hidden_features_BF = self._dropout(hidden_features_BF)
        binding_affinity_BO = self.layers[-1](hidden_features_BF)
        return binding_affinity_BO
