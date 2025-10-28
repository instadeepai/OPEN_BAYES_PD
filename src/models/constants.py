# pylint: disable=C0415, C0301

import pyro
from pyro.infer.autoguide import (
    AutoDiagonalNormal,
    AutoLowRankMultivariateNormal,
    AutoMultivariateNormal,
    AutoNormal,
)
from torch.optim import Adam, AdamW, RMSprop

from src.models.core.bayesian_cnn import BayesianCNN, HybridBayesianCNN
from src.models.core.bayesian_mlp import BayesianMLP
from src.models.core.bayesian_transformers import BayesianTransformer
from src.models.core.double_tower_bayesian_cnn import DoubleCNNWithNegative
from src.models.core.model_config import CNNConfig, MLPConfig, TransformerConfig
from src.models.core.quadruple_tower_bayesian_cnn import CNNWithBoltzmann

CNN_ARCHI: list[str] = [
    "ResNext",
    "CNN",
    "DoubleCNNWithNegative",
    "CNNWithBoltzmann",
    "HybridCNN",
]

MODEL_CONFIG_DICT = {
    "MLP": {
        "config": MLPConfig,
        "model": BayesianMLP,
    },
    "CNN": {
        "config": CNNConfig,
        "model": BayesianCNN,
    },
    "Transformer": {
        "config": TransformerConfig,
        "model": BayesianTransformer,
    },
    "DoubleCNNWithNegative": {
        "config": CNNConfig,
        "model": DoubleCNNWithNegative,
    },
    "CNNWithBoltzmann": {
        "config": CNNConfig,
        "model": CNNWithBoltzmann,
    },
    "HybridCNN": {
        "config": CNNConfig,
        "model": HybridBayesianCNN,
    },
}

optimizer_dict = {
    "AdamW": AdamW,
    "Adam": Adam,
    "RMSprop": RMSprop,
}

MODEL_PATH: str = "model.pth"

PARAMETERS_PATH: str = "params.pth"

PATH_DICT: dict[str, str] = {
    "model": MODEL_PATH,
    "parameters": PARAMETERS_PATH,
}

NUM_SAMPLES: int = 50

NUMERICAL_STABILITY_CONSTANT: int = 10**-8

GUIDE_TYPE: dict[str, pyro.infer.autoguide.AutoGuide] = {
    "MultivariateNormal": AutoMultivariateNormal,
    "DiagonalNormal": AutoDiagonalNormal,
    "Normal": AutoNormal,
    "LowRankMultivariateNormal": AutoLowRankMultivariateNormal,
}
