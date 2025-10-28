# pylint: disable=W0611
"""Configuration classes for models to reduce argument count."""

from dataclasses import dataclass
from typing import Optional

import torch
from torch.nn import AvgPool1d, BatchNorm1d, Identity, MaxPool1d
from torch.nn.functional import leaky_relu, relu, sigmoid, tanh


@dataclass
class CNNConfig:  # pylint: disable=too-many-instance-attributes
    """Configuration for CNN architecture.

    Parameters:
        list_of_channels: list of channels for each layer
        list_of_kernel_size: list of kernel sizes for each layer
        list_of_stride: list of strides for each layer
        layers_size: list of sizes for each layer
        prior_scale: prior scale for the weights
        activation_name: name of the activation function
        pooling_name: name of the pooling function
        batch_norm_name: name of the batch normalization function
        pool_size: list of pool sizes for each layer
        pool_stride: list of pool strides for each layer
        dropout_value: dropout value for each layer
        residual_connections: whether to use residual connections
        device: device to use for the model
    """

    list_of_channels: list[int]
    list_of_kernel_size: list[int]
    list_of_stride: list[int]
    layers_size: list[int]
    prior_scale: float
    activation_name: str
    pooling_name: str
    batch_norm_name: str
    pool_size: list[int]
    pool_stride: list[int]
    dropout_value: float
    list_of_padding: list[int]
    residual_connections: bool = False
    residual_skip_layers: int = 1
    device: Optional[torch.device] = None
    number_of_models: str = "distinct"


@dataclass
class MLPConfig:
    """Configuration for MLP architecture.

    Parameters:
    layers_size: list of sizes for each layer
    prior_scale: prior scale for the weights
    activation_name: name of the activation function
    dropout_value: dropout value for each layer
    device: device to use for the model
    """

    layers_size: list[int]
    prior_scale: float
    activation_name: str
    dropout_value: float
    device: Optional[torch.device] = None


@dataclass
class TransformerConfig:  # pylint: disable=too-many-instance-attributes
    """Configuration for Transformer architecture.

    Parameters:
        num_embeddings: input feature dimension per token
        sequence_length: input sequence length
        d_model: transformer model dimension
        nhead: number of attention heads
        num_encoder_layers: number of encoder layers
        dim_feedforward: feedforward dimension in transformer
        activation_name: activation function name for output head
        ffn_activation_name: activation function name for FFN inside encoder
        dropout_value: dropout probability
        prior_scale: prior scale for Bayesian weights
        layer_norm_eps: epsilon for layer norm stability
        batch_first: whether input tensors are (batch, seq, feature)
        pooling_type: 'mean' or 'cls'
        use_torch_mha: use nn.MultiheadAttention instead of manual projections
        device: torch device
    """

    num_embeddings: int
    sequence_length: int
    d_model: int
    nhead: int
    num_encoder_layers: int
    dim_feedforward: int
    activation_name: str
    ffn_activation_name: str = "ReLU"
    dropout_value: float = 0.1
    prior_scale: float = 2.0
    layer_norm_eps: float = 1e-5
    batch_first: bool = True
    pooling_type: str = "mean"
    device: Optional[torch.device] = None


@dataclass
class ActivationClass:  # pylint: disable=too-few-public-methods
    """Dictionary of activation functions.

    Parameters:
        tanh: Hyperbolic tangent function
        sigmoid: Sigmoid function
        relu: Rectified Linear Unit function
        leaky_relu: Leaky Rectified Linear Unit function
    """

    tanh = tanh
    sigmoid = sigmoid
    relu = relu
    leaky_relu = leaky_relu
    #  Aliases for common naming patterns
    ReLU = relu
    LeakyReLU = leaky_relu


@dataclass
class BatchNormalizationClass:  # pylint: disable=too-few-public-methods
    """Dictionary of batch normalization functions.

    Parameters:
        batch_norm: BatchNorm1d function
        none: Identity function
    """

    batch_norm = BatchNorm1d
    none = Identity

    # Aliases for common naming patterns
    BatchNorm1d = batch_norm


@dataclass
class PoolingClass:  # pylint: disable=too-few-public-methods
    """Dictionary of pooling functions.

    Parameters:
        max_pool_1d: MaxPool1d function
        avg_pool_1d: AvgPool1d function
        none: Identity function
    """

    max_pool_1d = MaxPool1d
    avg_pool_1d = AvgPool1d
    none = Identity

    # Aliases for common naming patterns
    MaxPool1d = max_pool_1d
    AvgPool1d = avg_pool_1d
