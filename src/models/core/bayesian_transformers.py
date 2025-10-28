"""
Transformer-based Bayesian sequence encoder predicting binding affinity.

Shape conventions:
- B: batch size
- L: sequence length
- E: embedding size per token
- O: output dimension (1 by default)
"""

# pylint: disable=R0913

from typing import Callable, Optional

import numpy as np
import pyro.distributions as dist
import torch
from pyro.nn import PyroModule, PyroSample
from torch import nn

from src.models.core.bayesian_mlp import get_activation
from src.models.core.model_config import TransformerConfig


# pylint: disable=too-many-instance-attributes
class BayesianTransformer(PyroModule):
    """Bayesian Transformer encoder followed by a linear head.

    The attention and feedforward linear projections are Bayesian via PyroSample.
    """

    def __init__(self, cfg: TransformerConfig):
        super().__init__()

        self._cfg = cfg
        self._activation = get_activation(cfg.activation_name)
        self._ffn_activation = get_activation(cfg.ffn_activation_name)
        self._dropout = nn.Dropout(cfg.dropout_value)
        self._positional_encoding = _PositionalEncoding(
            d_model=cfg.d_model,
            dropout=cfg.dropout_value,
            max_len=cfg.sequence_length,
            device=cfg.device,
        )

        # Embedding projection from input embeddings to d_model
        self.input_proj = PyroModule[nn.Linear](cfg.num_embeddings, cfg.d_model)

        encoder_layer = _BayesianTransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout_value,
            prior_scale=cfg.prior_scale,
            layer_norm_eps=cfg.layer_norm_eps,
            ffn_activation=self._ffn_activation,
            device=cfg.device,
        )
        self.encoder_layers = PyroModule[nn.ModuleList](
            [encoder_layer for _ in range(cfg.num_encoder_layers)]
        )
        self.layer_norm = nn.LayerNorm(cfg.d_model, eps=cfg.layer_norm_eps)

        # Pooling: mean over sequence
        self.pool = nn.AdaptiveAvgPool1d(1)

        # dense

        self.dense = PyroModule[nn.Linear](cfg.d_model, cfg.d_model)

        # Output head
        self.output_head = PyroModule[nn.Linear](cfg.d_model, 1)

        # Initialize Bayesian priors
        self._initialize_priors()

    def _initialize_priors(self) -> None:
        mean: float | torch.Tensor = 0.0
        if self._cfg.device is not None:
            mean = torch.tensor(0.0, device=self._cfg.device)

        # Input projection
        glorot_ip = self._cfg.num_embeddings + self._cfg.d_model
        self.input_proj.weight = PyroSample(
            dist.Normal(mean, self._cfg.prior_scale * np.sqrt(2 / glorot_ip))
            .expand([self.input_proj.out_features, self.input_proj.in_features])
            .to_event(self.input_proj.weight.dim())
        )
        self.input_proj.bias = PyroSample(
            dist.Normal(mean, self._cfg.prior_scale)
            .expand([self.input_proj.out_features])
            .to_event(self.input_proj.bias.dim())
        )
        glorot_oh = self._cfg.d_model + 1
        self.output_head.weight = PyroSample(
            dist.Normal(mean, self._cfg.prior_scale * np.sqrt(2 / glorot_oh))
            .expand([self.output_head.out_features, self.output_head.in_features])
            .to_event(self.output_head.weight.dim())
        )
        self.output_head.bias = PyroSample(
            dist.Normal(mean, self._cfg.prior_scale)
            .expand([self.output_head.out_features])
            .to_event(self.output_head.bias.dim())
        )
        glorot_dense = self._cfg.d_model + self._cfg.d_model
        self.dense.weight = PyroSample(
            dist.Normal(mean, self._cfg.prior_scale * np.sqrt(2 / glorot_dense))
            .expand([self.dense.out_features, self.dense.in_features])
            .to_event(self.dense.weight.dim())
        )
        self.dense.bias = PyroSample(
            dist.Normal(mean, self._cfg.prior_scale)
            .expand([self.dense.out_features])
            .to_event(self.dense.bias.dim())
        )

    def forward(self, input_embeddings_BLE: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            input_embeddings_BLE: tensor of shape (batch, length, embedding)

        Returns:
            binding_affinity_BO: tensor of shape (batch, 1)
        """
        if self._cfg.device is not None:
            input_embeddings_BLE = input_embeddings_BLE.to(self._cfg.device)

        x_BLE = input_embeddings_BLE
        x_BLE = self.input_proj(x_BLE)
        x_BLE = self._positional_encoding(x_BLE)

        # Apply stacked Bayesian encoder layers
        for encoder_layer in self.encoder_layers:
            x_BLE = encoder_layer(x_BLE)

        # x_BLE = self.layer_norm(x_BLE)

        # Pool across sequence length or take CLS-like token
        if self._cfg.pooling_type == "cls":
            x_BD = x_BLE[:, 0, :]
        else:
            x_BDL = x_BLE.transpose(1, 2)  # (B, D, L)
            x_BD1 = self.pool(x_BDL)
            x_BD = x_BD1.squeeze(-1)

        x_BD = self.dense(x_BD)
        x_BD = self._activation(x_BD)
        x_BD = self._dropout(x_BD)
        binding_affinity_BO = self.output_head(x_BD)
        return binding_affinity_BO


# pylint: disable=too-many-instance-attributes
class _BayesianTransformerEncoderLayer(PyroModule):
    """Single Bayesian Transformer encoder layer with MHA and FFN.
    All linear projections are Bayesian.
    """

    def __init__(  # pylint: disable=too-many-positional-arguments
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float,
        prior_scale: float,
        layer_norm_eps: float = 1e-5,
        ffn_activation: Callable[[torch.Tensor], torch.Tensor] = torch.relu,
        device: Optional[torch.device] = None,
    ):
        super().__init__()

        self._d_model = d_model
        self._nhead = nhead
        self._prior_scale = prior_scale
        self._device = device
        self._ffn_activation = ffn_activation

        # Single manual Bayesian MHA implementation
        self.q_proj = PyroModule[nn.Linear](d_model, d_model)
        self.k_proj = PyroModule[nn.Linear](d_model, d_model)
        self.v_proj = PyroModule[nn.Linear](d_model, d_model)
        self.out_proj = PyroModule[nn.Linear](d_model, d_model)

        self.attn_dropout = nn.Dropout(dropout)
        self.attn_layer_norm = nn.LayerNorm(d_model, eps=layer_norm_eps)

        # Feedforward
        self.ff1 = PyroModule[nn.Linear](d_model, dim_feedforward)
        self.ff2 = PyroModule[nn.Linear](dim_feedforward, d_model)
        self.ff_dropout = nn.Dropout(dropout)
        self.ff_layer_norm = nn.LayerNorm(d_model, eps=layer_norm_eps)

        # Initialize Bayesian priors on all linear layers
        self.layers = PyroModule[torch.nn.ModuleList](
            [self.q_proj, self.k_proj, self.v_proj, self.out_proj, self.ff1, self.ff2]
        )
        self._init_bayesian_linears()

    def _init_bayesian_linears(self) -> None:
        mean: float | torch.Tensor = 0.0
        if self._device is not None:
            mean = torch.tensor(0.0, device=self._device)

        def set_linear_prior(module: nn.Linear, fan_in: int, fan_out: int) -> None:
            if module:
                glorot = fan_in + fan_out
                module.weight = PyroSample(
                    dist.Normal(mean, self._prior_scale * np.sqrt(2 / glorot))
                    .expand([module.out_features, module.in_features])
                    .to_event(module.weight.dim())
                )
                module.bias = PyroSample(
                    dist.Normal(mean, self._prior_scale)
                    .expand([module.out_features])
                    .to_event(module.bias.dim())
                )

        # Q, K, V projections and output projection
        set_linear_prior(self.q_proj, self._d_model, self._d_model)
        set_linear_prior(self.k_proj, self._d_model, self._d_model)
        set_linear_prior(self.v_proj, self._d_model, self._d_model)
        set_linear_prior(self.out_proj, self._d_model, self._d_model)

        # Feedforward layers
        set_linear_prior(self.ff1, self._d_model, self.ff1.out_features)
        set_linear_prior(self.ff2, self.ff1.out_features, self._d_model)

    def forward(self, x_BLE: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        b, seq_len, d_model = x_BLE.size(0), x_BLE.size(1), x_BLE.size(2)
        # Self-attention with scaling (manual Bayesian)
        q = self.q_proj(x_BLE)
        k = self.k_proj(x_BLE)
        v = self.v_proj(x_BLE)

        # Reshape for multi-head: (B, L, D) -> (B, H, L, Dh)
        h = self._nhead
        dh = d_model // h
        q = q.view(b, seq_len, h, dh).transpose(1, 2)  # (B, H, L, Dh)
        k = k.view(b, seq_len, h, dh).transpose(1, 2)
        v = v.view(b, seq_len, h, dh).transpose(1, 2)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(dh)
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        attn_output = torch.matmul(attn_weights, v)  # (b, H, L, Dh)

        # Merge heads: (B, H, L, Dh) -> (B, L, D)
        attn_output = attn_output.transpose(1, 2).contiguous().view(b, seq_len, d_model)
        attn_output = self.out_proj(attn_output)

        # Residual + norm
        x_BLE = self.attn_layer_norm(x_BLE + attn_output)

        # Feedforward
        y = self.ff1(x_BLE)
        y = self._ffn_activation(y)
        y = self.ff_dropout(y)
        y = self.ff2(y)

        # Residual + norm
        x_BLE = self.ff_layer_norm(x_BLE + y)
        return x_BLE


class _PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding added to inputs."""

    def __init__(
        self,
        d_model: int,
        dropout: float,
        max_len: int,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, L, D)
        if device is not None:
            pe = pe.to(device)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x_BLE: torch.Tensor) -> torch.Tensor:
        """Forward pass of the positional encoding."""
        length = x_BLE.size(1)
        x_BLE = x_BLE + self.pe[:, :length]
        return self.dropout(x_BLE)
