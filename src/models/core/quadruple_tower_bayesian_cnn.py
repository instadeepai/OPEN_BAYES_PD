"""
Quadruple Tower Bayesian CNN implementation for phage display modeling.

This module contains the CNNWithBoltzmann class which implements a four-tower
architecture for phage display modeling. The four towers compute:
1. logits_binding_base
2. logits_non_binding_base
3. logits_binding_target
4. logits_non_binding_target

Architecture modes:
- number_of_models='one': 1 model with 4 outputs
- number_of_models='two': 2 models with 2 outputs each
- number_of_models='distinct': 4 separate models with 1 output each
"""

import copy
from typing import Tuple

import torch
from pyro.nn import PyroModule

from src.models.core.bayesian_cnn import BayesianCNN
from src.models.core.model_config import CNNConfig


class CNNWithBoltzmann(PyroModule):
    """
    Quadruple tower Bayesian CNN.

    Architecture depends on number_of_models parameter:
    - If number_of_models='one': 1 model with 4 outputs
    - If number_of_models='two': 2 models with 2 outputs each
    - Otherwise: 4 separate models with 1 output each

    Parameters:
    ----------
    cfg (CNNConfig):
        Configuration for the quadruple tower BayesianCNN.
    """

    def __init__(self, cfg: CNNConfig):
        """
        Initialize the quadruple tower Bayesian CNN.

        Parameters:
        ----------
        cfg (CNNConfig):
            Configuration for the quadruple tower BayesianCNN.
        """
        super().__init__()
        self.number_of_models = cfg.number_of_models

        match self.number_of_models:
            case "one":
                # Create 1 model with 4 outputs
                cfg_quad_output = self._modify_config_for_quad_output(cfg)
                self.target_model = BayesianCNN(cfg_quad_output)

                # Set all other models to None
                self.base_model = None
                self.base_non_binding_model = None
                self.target_non_binding_model = None

            case "two":
                # Create 2 models with 2 outputs each
                cfg_double_output = self._modify_config_for_double_output(cfg)
                self.base_model = BayesianCNN(cfg_double_output)
                self.target_model = BayesianCNN(cfg_double_output)

                # Set other models to None
                self.base_non_binding_model = None
                self.target_non_binding_model = None

            case "distinct":
                # Create four separate models with single outputs each
                self.base_model = BayesianCNN(cfg)
                self.base_non_binding_model = BayesianCNN(cfg)
                self.target_model = BayesianCNN(cfg)
                self.target_non_binding_model = BayesianCNN(cfg)

            case _:
                raise ValueError(
                    f"Invalid number_of_models value: {self.number_of_models}"
                )

    def _modify_config_for_quad_output(self, cfg: CNNConfig) -> CNNConfig:
        """
        Modify the configuration for quad output mode (4 outputs from 1 model).

        Parameters:
        ----------
        cfg (CNNConfig):
            Original configuration.

        Returns:
        -------
        CNNConfig:
            Modified configuration with 4 outputs.
        """
        modified_cfg = copy.deepcopy(cfg)
        # Change the last layer to have 4 outputs
        modified_cfg.layers_size[-1] = 4
        return modified_cfg

    def _modify_config_for_double_output(self, cfg: CNNConfig) -> CNNConfig:
        """
        Modify the configuration for double output mode (2 outputs per model).

        Parameters:
        ----------
        cfg (CNNConfig):
            Original configuration.

        Returns:
        -------
        CNNConfig:
            Modified configuration with 2 outputs.
        """
        modified_cfg = copy.deepcopy(cfg)
        # Change the last layer to have 2 outputs
        modified_cfg.layers_size[-1] = 2
        return modified_cfg

    def _forward_quad_output(
        self, data: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass using single model with 4 outputs (number_of_models='one').

        Parameters:
        ----------
        data (torch.Tensor):
            Input data tensor.

        Returns:
        -------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            logits_binding_base, logits_non_binding_base,
            logits_binding_target, logits_non_binding_target
        """
        # Single model outputs: [binding_base, non_binding_base,
        #                       binding_target, non_binding_target]
        quad_outputs = self.target_model(data)
        logits_binding_base = quad_outputs[:, 0:1]
        logits_non_binding_base = quad_outputs[:, 1:2]
        logits_binding_target = quad_outputs[:, 2:3]
        logits_non_binding_target = quad_outputs[:, 3:4]

        return (
            logits_binding_base,
            logits_non_binding_base,
            logits_binding_target,
            logits_non_binding_target,
        )

    def _forward_double_output(
        self, data: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass using double output models (number_of_models='two').

        Parameters:
        ----------
        data (torch.Tensor):
            Input data tensor.

        Returns:
        -------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            logits_binding_base, logits_non_binding_base,
            logits_binding_target, logits_non_binding_target
        """
        # base model outputs: [binding_base, non_binding_base]
        base_outputs = self.base_model(data)
        logits_binding_base = base_outputs[:, 0:1]
        logits_non_binding_base = base_outputs[:, 1:2]

        # Target model outputs: [binding_target, non_binding_target]
        target_outputs = self.target_model(data)
        logits_binding_target = target_outputs[:, 0:1]
        logits_non_binding_target = target_outputs[:, 1:2]

        return (
            logits_binding_base,
            logits_non_binding_base,
            logits_binding_target,
            logits_non_binding_target,
        )

    def _forward_single_output(
        self, data: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass using distinct models (number_of_models='distinct').

        Parameters:
        ----------
        data (torch.Tensor):
            Input data tensor.

        Returns:
        -------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            logits_binding_base, logits_non_binding_base,
            logits_binding_target, logits_non_binding_target
        """
        # Forward pass through each individual model
        logits_binding_base = self.base_model(data)
        logits_non_binding_base = self.base_non_binding_model(data)
        logits_binding_target = self.target_model(data)
        logits_non_binding_target = self.target_non_binding_model(data)

        return (
            logits_binding_base,
            logits_non_binding_base,
            logits_binding_target,
            logits_non_binding_target,
        )

    def forward(
        self, data: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the quadruple tower Bayesian CNN.

        Parameters:
        ----------
        data (torch.Tensor):
            Input data tensor with shape (batch_size, channels, sequence_length).

        Returns:
        -------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            - logits_binding_base: Logits for binding to base
            - logits_non_binding_base: Logits for non-binding to base
            - logits_binding_target: Logits for binding to target
            - logits_non_binding_target: Logits for non-binding to target
        """
        match self.number_of_models:
            case "one":
                return self._forward_quad_output(data)
            case "two":
                return self._forward_double_output(data)
            case "distinct":
                return self._forward_single_output(data)
            case _:
                raise ValueError(
                    f"Invalid number_of_models value: {self.number_of_models}"
                )
