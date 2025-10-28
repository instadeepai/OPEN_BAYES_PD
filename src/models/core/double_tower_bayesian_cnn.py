"""
Double Tower Bayesian CNN implementation for phage display modeling.

This module contains the DoubleCNNWithNegative class which implements a two-tower
architecture for phage display modeling. The first tower computes negative binding
probability and the second tower computes final binding affinity.
"""

import copy
from typing import Tuple

import torch
from pyro.nn import PyroModule

from src.models.core.bayesian_cnn import BayesianCNN
from src.models.core.model_config import CNNConfig


class DoubleCNNWithNegative(PyroModule):
    """
    Double tower Bayesian CNN.

    First model computes p binding neg, then second model uses both p binding neg
    and original data to compute final binding affinity. The p binding neg is used
    as input to the MLP part of the second model, not the CNN part.

    Parameters:
    ----------
    cfg (CNNConfig):
        Configuration for the double tower BayesianCNN.
    """

    def __init__(self, cfg: CNNConfig):
        """
        Initialize the double tower Bayesian CNN.

        Parameters:
        ----------
        cfg (CNNConfig):
            Configuration for the double tower BayesianCNN.
        """
        super().__init__()
        self.number_of_models = cfg.number_of_models

        # Configure models based on number_of_models mode
        if self.number_of_models == "one":
            # For 'one' mode, modify config to have 2 outputs
            cfg_one_model = self._modify_config_for_one_mode(cfg)
            self.first_model = BayesianCNN(cfg_one_model)
            self.second_model = None
        else:
            # First model: computes p binding neg
            self.first_model = BayesianCNN(cfg)
            # Second model: uses p binding neg + original data
            cfg_second_model = self._modify_second_model_config(cfg)
            self.second_model = BayesianCNN(cfg_second_model)

    def _modify_config_for_one_mode(self, cfg: CNNConfig) -> CNNConfig:
        """
        Modify the configuration for 'one' mode to have 2 outputs.

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
        # Change the last layer to have 2 outputs: p_binding_neg and binding_affinity
        modified_cfg.layers_size[-1] = 2
        return modified_cfg

    def _modify_second_model_config(self, cfg: CNNConfig) -> CNNConfig:
        """
        Modify the second model configuration to account for the additional
        p binding neg input in the MLP part.

        Parameters:
        ----------
        cfg (CNNConfig):
            Original configuration for the second model.

        Returns:
        -------
        CNNConfig:
            Modified configuration with adjusted MLP input size.
        """
        # Create a copy of the config to avoid modifying the original
        modified_cfg = copy.deepcopy(cfg)

        # Get the output dimension of the first model (number of output neurons)
        if self.number_of_models == "sequential_latent":
            first_model_output_dim = cfg.layers_size[-2]
        elif self.number_of_models == "sequential_output":
            first_model_output_dim = cfg.layers_size[-1]
        elif self.number_of_models == "distinct":
            return cfg
        elif self.number_of_models == "one":
            first_model_output_dim = cfg.layers_size[-1] + 1
        else:
            raise ValueError(f"Invalid number_of_models value: {self.number_of_models}")

        # Modify the first MLP layer size to include the first model output
        # The first MLP layer receives flattened conv output + first model output
        modified_cfg.layers_size[0] += first_model_output_dim

        return modified_cfg

    def _forward_sequential_latent(
        self, data: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass using sequential latent representation."""
        (
            logits_binding_neg,
            latent_representation,
        ) = self.first_model.forward_with_latent(data)
        first_model_features = latent_representation

        # Forward through conv layers of second model only
        layer_outputs = {} if self.second_model.use_residual else None
        representation = self.second_model.forward_conv_layer(data, 0, layer_outputs)

        for layer_idx in range(1, self.second_model.number_of_conv_modules):
            representation = self.second_model.forward_conv_layer(
                representation, layer_idx, layer_outputs
            )

        # Flatten the conv output
        representation = representation.reshape(representation.shape[0], -1)

        # Concatenate conv output with latent representation from first model
        combined_representation = torch.cat(
            [representation, first_model_features], dim=1
        )

        # Forward through MLP layers of second model
        num_layers = len(self.second_model.layers)
        for layer_idx in range(
            self.second_model.number_of_conv_modules, num_layers - 1
        ):
            # Use combined representation for the first MLP layer
            if layer_idx == self.second_model.number_of_conv_modules:
                combined_representation = self.second_model.layers[layer_idx](
                    combined_representation
                )
                combined_representation = self.second_model.activation(
                    combined_representation
                )
                combined_representation = self.second_model.dropout(
                    combined_representation
                )
            else:
                combined_representation = self.second_model.forward_mlp_layer(
                    combined_representation, layer_idx
                )

        # Final layer
        logits_binding_pos = self.second_model.layers[-1](combined_representation)

        return logits_binding_neg, logits_binding_pos

    def _forward_distinct(
        self, data: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass using distinct models."""
        logits_binding_neg = self.first_model(data)
        logits_binding_pos = self.second_model(data)
        return logits_binding_neg, logits_binding_pos

    def _forward_sequential_output(
        self, data: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass using sequential output."""
        logits_binding_neg = self.first_model(data)
        first_model_features = logits_binding_neg

        # Forward through conv layers of second model only
        layer_outputs = {} if self.second_model.use_residual else None
        representation = self.second_model.forward_conv_layer(data, 0, layer_outputs)

        for layer_idx in range(1, self.second_model.number_of_conv_modules):
            representation = self.second_model.forward_conv_layer(
                representation, layer_idx, layer_outputs
            )

        # Flatten the conv output
        representation = representation.reshape(representation.shape[0], -1)

        # Concatenate conv output with output from first model
        combined_representation = torch.cat(
            [representation, first_model_features], dim=1
        )

        # Forward through MLP layers of second model
        num_layers = len(self.second_model.layers)
        for layer_idx in range(
            self.second_model.number_of_conv_modules, num_layers - 1
        ):
            # Use combined representation for the first MLP layer
            if layer_idx == self.second_model.number_of_conv_modules:
                combined_representation = self.second_model.layers[layer_idx](
                    combined_representation
                )
                combined_representation = self.second_model.activation(
                    combined_representation
                )
                combined_representation = self.second_model.dropout(
                    combined_representation
                )
            else:
                combined_representation = self.second_model.forward_mlp_layer(
                    combined_representation, layer_idx
                )

        # Final layer
        logits_binding_pos = self.second_model.layers[-1](combined_representation)

        return logits_binding_neg, logits_binding_pos

    def _forward_one(self, data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass using single model with 2 outputs."""
        # The first model now has 2 outputs: p_binding_neg and binding_affinity
        output = self.first_model(data)
        logits_binding_neg = output[:, 0:1]  # First output
        logits_binding_pos = output[:, 1:2]  # Second output
        return logits_binding_neg, logits_binding_pos

    def forward(self, data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the double tower Bayesian CNN.

        Parameters:
        ----------
        data (torch.Tensor):
            Input data tensor with shape (batch_size, channels, sequence_length).

        Returns:
        -------
        Tuple[torch.Tensor, torch.Tensor]:
            P binding neg from first model and final binding affinity.
        """
        # Call appropriate forward method based on number_of_models configuration
        match self.number_of_models:
            case "sequential_latent":
                return self._forward_sequential_latent(data)
            case "distinct":
                return self._forward_distinct(data)
            case "sequential_output":
                return self._forward_sequential_output(data)
            case "one":
                return self._forward_one(data)
            case _:
                raise ValueError(
                    f"Invalid number_of_models value: {self.number_of_models}"
                )
