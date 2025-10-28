import random

import numpy as np
import pyro
import torch

SEED_FOR_RNG: int = 42


def set_seed():
    """Set RNG seed for python and the libraries involved in training"""
    np.random.seed(SEED_FOR_RNG)
    random.seed(SEED_FOR_RNG)
    torch.manual_seed(SEED_FOR_RNG)
    pyro.set_rng_seed(SEED_FOR_RNG)
