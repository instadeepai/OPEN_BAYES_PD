"""
Shape Suffixes Documentation:

B: batch size
C: channels (number of channels in convolutional layers)
L: sequence length (length of the input sequence)
F: features (hidden dimensions in MLP layers)
O: output (final output dimension)

Example:
- input_embeddings_BLC: input embeddings with batch, length, and channel dimensions
"""

# pylint: disable=R0913,R0917

from typing import Optional

import numpy as np
import pyro.distributions as dist
import torch
from pyro.nn import PyroModule, PyroSample
from torch import nn

from src.models.core.bayesian_mlp import get_activation, get_batch_norm
from src.models.core.model_config import CNNConfig, PoolingClass


class IdentityWrapper:
    """Identity wrapper that mimics PyroModule behavior for deterministic layers."""

    def __getitem__(self, module_class):
        """Return the module class as-is without any wrapping."""
        return module_class

    def __call__(self, *args, **kwargs):
        """Allow calling the wrapper (for compatibility)."""
        return self


def get_pooling(pooling_name: str) -> nn.Module:
    """
    Get the pooling function.
    """
    if pooling_name in PoolingClass.__dict__:
        return PoolingClass.__dict__[pooling_name]

    raise NotImplementedError(f"Pooling function '{pooling_name}' not implemented")


# TODO let's use gain and then modify and move around the get_activation function


class BayesianCNN(PyroModule):  # pylint: disable=too-many-instance-attributes
    """
    CNN consisting in conv1D and linear layers wrapped in PyroModule allowing Bayesian
    training of the weights. SO this is more than a CNN module as it also contains a
    downstream task MLP

    Parameters:
    ----------
    list_channels (list[int]):
        contains the number of channels for the whole network idx i and idx i+1 will
        create particular CNN layer
    list_kernel_size (list[int]):
        contains the kernel size of the 1D conv used in the whole network, where
        kernel_size idx i refers to conv1D layer i
    list_stride (list[int]):
        contains the stride size of the 1D conv used in the whole network, where stride
        idx i refers to conv1D layer i
    layers_size (list[int]):
        size of input, hidden and output layers for the MLP part.
    prior_scale (float):
        variance (uncertainty) of the prior distribution over the network's weights and
        biases.
    activation_name (str):
        name of the activation function for the MLP part.
    device (Optional[torch.device]):
        define the device, cuda or cpu, to work with.
    """

    def __init__(
        self,
        cfg: CNNConfig,
    ):
        """
        Constructor for the CNN.
        """
        super().__init__()

        # Handle single value parameters before initializing layers
        cfg = self._handle_single_value_parameters(cfg)

        self.activation = get_activation(cfg.activation_name)
        pooling_class = get_pooling(cfg.pooling_name)
        self.poolings = [
            pooling_class(kernel_size=cfg.pool_size[i], stride=cfg.pool_stride[i])
            for i in range(len(cfg.pool_size))
        ]
        self.batch_norm = get_batch_norm(cfg.batch_norm_name)
        self.pool_sizes = cfg.pool_size
        self.pool_strides = cfg.pool_stride
        self.dropout = nn.Dropout(cfg.dropout_value)
        self.use_residual = cfg.residual_connections
        self.residual_skip_layers = cfg.residual_skip_layers
        self.strides = cfg.list_of_stride
        self.number_of_models = cfg.number_of_models

        # Initialize attributes that will be set during layer initialization
        self.number_of_conv_modules = 0
        self.batch_norm_layers = None
        self.projection_layers = None
        self.residual_mappings = {}

        assert np.all(cfg.list_of_channels) > 0
        self._initialize_layers(cfg)

    def _initialize_layers(self, cfg: CNNConfig):
        """Initialize all layers by calling specialized initialization methods.

        Parameters:
        ----------
        cfg (CNNConfig):
            configuration for the CNN.

        Returns:
        -------
        None
        """
        # BayesianCNN uses PyroModules
        wrapper = PyroModule
        list_wrapper = PyroModule[torch.nn.ModuleList]
        conv_layers = self._initialize_conv_layers(cfg, wrapper)
        self._initialize_batch_norm_layers(cfg, wrapper, list_wrapper)
        mlp_layers = self._initialize_mlp_layers(cfg, wrapper)
        self._initialize_projection_layers(cfg, wrapper, list_wrapper)

        # Combine conv and MLP layers
        self.layers = PyroModule[torch.nn.ModuleList](conv_layers + mlp_layers)

        # Initialize weights
        self._initialize_weights(
            cfg.list_of_channels,
            cfg.list_of_kernel_size,
            cfg.layers_size,
            cfg.prior_scale,
            cfg.device,
        )

    def _initialize_conv_layers(self, cfg: CNNConfig, wrapper=PyroModule) -> list:
        """Initialize convolutional layers.

        Parameters:
        ----------
        cfg (CNNConfig):
            configuration for the CNN.
        wrapper:
            wrapper function (PyroModule or identity function)

        Returns:
        -------
        list: List of convolutional layers
        """
        layer_list_conv = [
            wrapper[nn.Conv1d](
                in_channels=cfg.list_of_channels[idx - 1],
                out_channels=cfg.list_of_channels[idx],
                kernel_size=cfg.list_of_kernel_size[idx - 1],
                stride=cfg.list_of_stride[idx - 1],
                padding=cfg.list_of_padding[idx - 1],
            )
            if cfg.list_of_kernel_size[idx - 1] > 0
            else nn.Identity()
            for idx in range(1, len(cfg.list_of_channels))
        ]

        self.number_of_conv_modules = len(layer_list_conv)
        return layer_list_conv

    def _initialize_batch_norm_layers(
        self, cfg: CNNConfig, wrapper=PyroModule, list_wrapper=None
    ):
        """Initialize batch normalization layers.

        Parameters:
        ----------
        cfg (CNNConfig):
            configuration for the CNN.
        wrapper:
            wrapper function (PyroModule or identity function)
        list_wrapper:
            wrapper for ModuleList
        """
        if list_wrapper is None:
            list_wrapper = PyroModule[torch.nn.ModuleList]

        self.batch_norm_layers = list_wrapper(
            [
                wrapper[self.batch_norm](cfg.list_of_channels[idx])
                for idx in range(1, len(cfg.list_of_channels))
            ]
        )

    def _initialize_mlp_layers(self, cfg: CNNConfig, wrapper=PyroModule) -> list:
        """Initialize MLP layers.

        Parameters:
        ----------
        cfg (CNNConfig):
            configuration for the CNN.
        wrapper:
            wrapper function (PyroModule or identity function)

        Returns:
        -------
        list: List of MLP layers
        """
        layers_size = cfg.layers_size.copy()
        if cfg.number_of_models == "one":
            layers_size[-1] = layers_size[-1] + 1
        return [
            wrapper[nn.Linear](layers_size[idx - 1], layers_size[idx])
            for idx in range(1, len(layers_size))
        ]

    def _initialize_projection_layers(
        self, cfg: CNNConfig, wrapper=PyroModule, list_wrapper=None
    ):
        """Initialize projection layers for residual connections.

        Parameters:
        ----------
        cfg (CNNConfig):
            configuration for the CNN.
        wrapper:
            wrapper function (PyroModule or identity function)
        list_wrapper:
            wrapper for ModuleList
        """
        if not self.use_residual:
            self.projection_layers = None
            return

        if list_wrapper is None:
            list_wrapper = PyroModule[torch.nn.ModuleList]

        projection_layers_list = []
        self.residual_mappings = {}  # Track which layers have residual connections

        for input_layer_idx in range(len(cfg.list_of_channels) - 1):
            target_layer_idx = input_layer_idx + cfg.residual_skip_layers

            # Check if target layer exists
            if target_layer_idx < len(cfg.list_of_channels) - 1:
                self.residual_mappings[target_layer_idx] = input_layer_idx

                # Calculate cumulative stride for skipped layers
                cumulative_stride = 1
                for skip_idx in range(input_layer_idx, target_layer_idx):
                    cumulative_stride *= cfg.list_of_stride[skip_idx]

                # Create projection layer
                needs_projection = (
                    cfg.list_of_channels[input_layer_idx]
                    != cfg.list_of_channels[target_layer_idx + 1]
                    or cumulative_stride != 1
                )
                if needs_projection:
                    projection_layer = wrapper[nn.Conv1d](
                        in_channels=cfg.list_of_channels[input_layer_idx],
                        out_channels=cfg.list_of_channels[target_layer_idx + 1],
                        kernel_size=1,  # Use 1x1 conv for projection
                        stride=cumulative_stride,
                        padding=0,  # No padding for 1x1 conv
                    )
                else:
                    projection_layer = nn.Identity()

                projection_layers_list.append(projection_layer)

        self.projection_layers = list_wrapper(projection_layers_list)

    def _handle_single_value_parameters(self, cfg: CNNConfig) -> CNNConfig:
        """Handle single value parameters by expanding them to match the number
        of conv layers.
        """
        # Number of conv layers = number of channels - 1
        num_conv_layers = len(cfg.list_of_channels) - 1
        if len(cfg.list_of_kernel_size) == 1 and num_conv_layers > 1:
            cfg.list_of_kernel_size = [cfg.list_of_kernel_size[0]] * num_conv_layers
        if len(cfg.list_of_stride) == 1 and num_conv_layers > 1:
            cfg.list_of_stride = [cfg.list_of_stride[0]] * num_conv_layers
        if len(cfg.pool_size) == 1 and num_conv_layers > 1:
            cfg.pool_size = [cfg.pool_size[0]] * num_conv_layers
        if len(cfg.pool_stride) == 1 and num_conv_layers > 1:
            cfg.pool_stride = [cfg.pool_stride[0]] * num_conv_layers
        if len(cfg.list_of_padding) == 1 and num_conv_layers > 1:
            cfg.list_of_padding = [cfg.list_of_padding[0]] * num_conv_layers
        return cfg

    def _initialize_weights(
        self,
        list_of_channels: list[int],
        list_of_kernel_size: list[int],
        layers_size: list[int],
        prior_scale: float,
        device: Optional[torch.device] = None,
    ):
        """Initialize weights and biases for all layers using Glorot/Xavier
        initialization.
        """
        mean = 0.0
        if device is not None:
            mean = torch.tensor(0.0, device=device)

        for layer_idx, layer in enumerate(self.layers):
            if layer_idx < self.number_of_conv_modules:
                # For convolutional layers: fan_in = in_channels * kernel_size,
                # fan_out = out_channels * kernel_size
                fan_in = list_of_channels[layer_idx] * list_of_kernel_size[layer_idx]
                fan_out = (
                    list_of_channels[layer_idx + 1] * list_of_kernel_size[layer_idx]
                )
                glorot = fan_in + fan_out
            else:
                # For linear layers: fan_in = input_size, fan_out = output_size
                mlp_layer_idx = layer_idx - self.number_of_conv_modules
                glorot = layers_size[mlp_layer_idx] + layers_size[mlp_layer_idx + 1]

            layer.weight = PyroSample(
                dist.Normal(mean, prior_scale * np.sqrt(2 / glorot))
                .expand(layer.weight.shape)
                .to_event(layer.weight.dim())
            )
            layer.bias = PyroSample(
                dist.Normal(mean, prior_scale)
                .expand(layer.bias.shape)
                .to_event(layer.bias.dim())
            )
        if self.batch_norm is not nn.Identity:
            self._initialize_batch_norm_weights(mean, prior_scale)

        # Initialize projection layers for residual connections
        if self.use_residual and self.projection_layers is not None:
            self._initialize_projection_weights(mean, prior_scale)

    def _initialize_batch_norm_weights(self, mean: float, prior_scale: float):
        """Initialize weights for the batch norm layers."""
        for layer in self.batch_norm_layers:
            layer.weight = PyroSample(
                dist.Normal(mean, prior_scale)
                .expand(layer.weight.shape)
                .to_event(layer.weight.dim())
            )
            layer.bias = PyroSample(
                dist.Normal(mean, prior_scale)
                .expand(layer.bias.shape)
                .to_event(layer.bias.dim())
            )

    def _initialize_projection_weights(self, mean: float, prior_scale: float):
        """Initialize weights for the projection layers used in residual connections."""
        for layer in self.projection_layers:
            if hasattr(layer, "weight") and hasattr(layer, "bias"):
                # Calculate fan_in and fan_out for projection layers (1x1 convolutions)
                fan_in = layer.weight.shape[1]  # input channels
                fan_out = layer.weight.shape[0]  # output channels
                glorot = fan_in + fan_out

                layer.weight = PyroSample(
                    dist.Normal(mean, prior_scale * np.sqrt(2 / glorot))
                    .expand(layer.weight.shape)
                    .to_event(layer.weight.dim())
                )
                layer.bias = PyroSample(
                    dist.Normal(mean, prior_scale)
                    .expand(layer.bias.shape)
                    .to_event(layer.bias.dim())
                )

    def forward_conv_layer(
        self, conv_input_BLC: torch.Tensor, layer_idx: int, layer_outputs: dict = None
    ) -> torch.Tensor:
        """
        Forward function: run convolutional layer over the input data.

        Parameters:
        ----------
        conv_input_BLC (torch.Tensor):
            input tensor representation with shape (batch, length, channels).
        layer_idx (int):
            index of the convolutional layer to apply.
        layer_outputs (dict):
            dictionary to store layer outputs for residual connections.

        Returns:
        -------
        torch.Tensor:
            transformed representation after conv, batch norm, activation,
            pooling and dropout with shape (batch, length, channels).
        """
        # Store input if this layer might be used for residual connection
        if layer_outputs is not None and self.use_residual:
            layer_outputs[layer_idx] = conv_input_BLC

        # Forward pass through conv layer
        conv_output_BLC = self.layers[layer_idx](conv_input_BLC)
        # Apply batch norm to the output channels of the current conv layer
        conv_output_BLC = self.batch_norm_layers[layer_idx](conv_output_BLC)

        # Add residual connection if this layer is a target for residual connection
        if (
            self.use_residual
            and layer_outputs is not None
            and layer_idx in self.residual_mappings
        ):
            source_layer_idx = self.residual_mappings[layer_idx]
            if source_layer_idx in layer_outputs:
                identity_BLC = layer_outputs[source_layer_idx]
                # Find corresponding projection layer
                proj_layer_idx = list(self.residual_mappings.keys()).index(layer_idx)
                identity_proj_BLC = self.projection_layers[proj_layer_idx](identity_BLC)

                # Only add residual if dimensions match after projection
                dimensions_match = (
                    identity_proj_BLC.shape[1] == conv_output_BLC.shape[1]
                    and identity_proj_BLC.shape[2] == conv_output_BLC.shape[2]
                )
                if dimensions_match:
                    conv_output_BLC = conv_output_BLC + identity_proj_BLC

        conv_output_BLC = self.activation(conv_output_BLC)
        conv_output_BLC = self.poolings[layer_idx](conv_output_BLC)
        conv_output_BLC = self.dropout(conv_output_BLC)
        return conv_output_BLC

    def forward_mlp_layer(
        self, mlp_input_BF: torch.Tensor, layer_idx: int
    ) -> torch.Tensor:
        """
        Forward function: run MLP layer over the input data.

        Parameters:
        ----------
        mlp_input_BF (torch.Tensor):
            input tensor representation with shape (batch, features).
        layer_idx (int):
            index of the MLP layer to apply.

        Returns:
        -------
        torch.Tensor:
            transformed representation with shape (batch, features).
        """
        mlp_output_BF = self.layers[layer_idx](mlp_input_BF)
        mlp_output_BF = self.activation(mlp_output_BF)
        mlp_output_BF = self.dropout(mlp_output_BF)
        return mlp_output_BF

    def forward_with_latent(self, input_embeddings_BLC: torch.Tensor) -> torch.Tensor:
        """
        Forward function: run BayesianCNN over the input data.

        Parameters:
        ----------
        input_embeddings_BLC (torch.Tensor):
            input data as embeddings of the sequence with shape
            (batch, length, channels).

        Returns:
        -------
        tuple:
            (binding_affinity_BO, latent_features_BF): unnormalized binding
            probability and latent representation.
        """
        layer_outputs = {} if self.use_residual else None
        conv_output_BLC = self.forward_conv_layer(
            input_embeddings_BLC, 0, layer_outputs
        )

        for layer_idx in range(1, self.number_of_conv_modules):
            conv_output_BLC = self.forward_conv_layer(
                conv_output_BLC, layer_idx, layer_outputs
            )

        # Flatten the convolutional output for MLP layers
        mlp_input_BF = conv_output_BLC.reshape(conv_output_BLC.shape[0], -1)

        for layer_idx in range(self.number_of_conv_modules, len(self.layers) - 1):
            mlp_input_BF = self.forward_mlp_layer(mlp_input_BF, layer_idx)

        binding_affinity_BO = self.layers[-1](mlp_input_BF)
        return binding_affinity_BO, mlp_input_BF

    def forward(self, input_embeddings_BLC: torch.Tensor) -> torch.Tensor:
        """
        Forward function: run BayesianCNN over the input data.

        Parameters:
        ----------
        input_embeddings_BLC (torch.Tensor):
            input data as embeddings of the sequence with shape
            (batch, length, channels).

        Returns:
        -------
        torch.Tensor:
            binding affinity with shape (batch, output).
        """
        return self.forward_with_latent(input_embeddings_BLC)[0]


class DeterministicBayesianCNNCounterpart(BayesianCNN):
    # pylint: disable=too-many-instance-attributes
    """
    Pytorch CNN mirroring the pyro BayesianCNN.

    Parameters:
    ----------
    list_channels (list[int]):
        contains the number of channels for the whole network idx i and idx i+1 will
        create particular CNN layer
    list_kernel_size (list[int]):
        contains the kernel size of the 1D conv used in the whole network, where
        kernel_size idx i refers to conv1D layer i
    list_stride (list[int]):
        contains the stride size of the 1D conv used in the whole network, where stride
        idx i refers to conv1D layer i
    layers_size (list[int]):
        size of input, hidden and output layers for the MLP part.
    activation_name (str):
        name of the activation function for the MLP part.
    """

    def _initialize_weights(
        self,
        list_of_channels: list[int],
        list_of_kernel_size: list[int],
        layers_size: list[int],
        prior_scale: float,
        device: Optional[torch.device] = None,
    ):
        """Initialize weights and biases for all layers using standard PyTorch
        initialization (for deterministic model).
        """
        # Empty implementation - use PyTorch default initialization

    def _initialize_layers(self, cfg: CNNConfig):
        """Initialize all layers by calling specialized initialization methods.

        Parameters:
        ----------
        cfg (CNNConfig):
            configuration for the CNN.

        Returns:
        -------
        None
        """
        # DeterministicBayesianCNNCounterpart uses regular PyTorch modules
        wrapper = IdentityWrapper()
        list_wrapper = torch.nn.ModuleList
        conv_layers = self._initialize_conv_layers(cfg, wrapper)
        self._initialize_batch_norm_layers(cfg, wrapper, list_wrapper)
        mlp_layers = self._initialize_mlp_layers(cfg, wrapper)
        self._initialize_projection_layers(cfg, wrapper, list_wrapper)

        # Combine conv and MLP layers
        self.layers = torch.nn.ModuleList(conv_layers + mlp_layers)


class HybridBayesianCNN(BayesianCNN):
    # pylint: disable=too-many-instance-attributes
    """
    Pytorch CNN mirroring the pyro BayesianCNN.

    Parameters:
    ----------
    list_channels (list[int]):
        contains the number of channels for the whole network idx i and idx i+1 will
        create particular CNN layer
    list_kernel_size (list[int]):
        contains the kernel size of the 1D conv used in the whole network, where
        kernel_size idx i refers to conv1D layer i
    list_stride (list[int]):
        contains the stride size of the 1D conv used in the whole network, where stride
        idx i refers to conv1D layer i
    layers_size (list[int]):
        size of input, hidden and output layers for the MLP part.
    activation_name (str):
        name of the activation function for the MLP part.
    """

    def _initialize_weights(
        self,
        list_of_channels: list[int],
        list_of_kernel_size: list[int],
        layers_size: list[int],
        prior_scale: float,
        device: Optional[torch.device] = None,
    ):
        """Initialize weights and biases for all layers.
        CNN layers will use PyTorch default initialization,
        MLP layers will use PyroSample for Bayesian initialization.
        """
        mean = 0.0
        if device is not None:
            mean = torch.tensor(0.0, device=device)

        # The MLP layers start after number_of_conv_modules
        for layer_idx in range(self.number_of_conv_modules, len(self.layers)):
            layer = self.layers[layer_idx]
            # For linear layers: fan_in = input_size, fan_out = output_size
            mlp_layer_relative_idx = layer_idx - self.number_of_conv_modules
            glorot = (
                layers_size[mlp_layer_relative_idx]
                + layers_size[mlp_layer_relative_idx + 1]
            )

            layer.weight = PyroSample(
                dist.Normal(mean, prior_scale * np.sqrt(2 / glorot))
                .expand(layer.weight.shape)
                .to_event(layer.weight.dim())
            )
            layer.bias = PyroSample(
                dist.Normal(mean, prior_scale)
                .expand(layer.bias.shape)
                .to_event(layer.bias.dim())
            )

    def _initialize_layers(self, cfg: CNNConfig):
        """Initialize all layers by calling specialized initialization methods.

        Parameters:
        ----------
        cfg (CNNConfig):
            configuration for the CNN.

        Returns:
        -------
        None
        """
        deterministic_wrapper = IdentityWrapper()
        probabilistic_wrapper = PyroModule
        list_wrapper = torch.nn.ModuleList  # For deterministic ModuleLists

        # Convolutional layers will be deterministic
        conv_layers = self._initialize_conv_layers(cfg, deterministic_wrapper)
        # Batch Norm layers for CNN will be deterministic
        self._initialize_batch_norm_layers(cfg, deterministic_wrapper, list_wrapper)
        # MLP layers will be probabilistic (PyroModule wrapped)
        mlp_layers = self._initialize_mlp_layers(cfg, probabilistic_wrapper)
        # Projection layers for residual connections will be deterministic
        self._initialize_projection_layers(cfg, deterministic_wrapper, list_wrapper)

        self.layers = PyroModule[torch.nn.ModuleList](conv_layers + mlp_layers)

        # Call the modified _initialize_weights to set priors for MLP layers
        self._initialize_weights(
            cfg.list_of_channels,
            cfg.list_of_kernel_size,
            cfg.layers_size,
            cfg.prior_scale,
            cfg.device,
        )
