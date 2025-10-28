from enum import Enum


class DatasetType(Enum):
    """Different types of what a dataset can be."""

    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


class ModelType(Enum):
    """Different types of what a model can be."""

    MLP = "MLP"
    CNN = "CNN"
    HYBRID_CNN = "HybridCNN"
    RESNEXT = "ResNext"
    DOUBLE_TOWER_CNN = "DoubleCNNWithNegative"
    QUADRUPLE_TOWER_CNN = "CNNWithBoltzmann"
    TRANSFORMER = "Transformer"
