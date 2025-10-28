"""
Configuration classes for model evaluation.
"""
from dataclasses import dataclass

import pyro
from neptune import Run as NeptuneRun
from pyro.nn import PyroModule


@dataclass
class EvaluationConfig:
    """Configuration for model evaluation.

    Parameters:
    -----------
    model (PyroModule):
        The probabilistic model which approximates the likelihood of the data.
    guide (pyro.infer.autoguide):
        The guide (approximate posterior) used for variational inference.
    neptune_run (NeptuneRun | None):
        Neptune object to load metrics.
    epoch (int):
        Current epoch number.
    every_n_epochs (int):
        Frequency of plot generation.
    """

    model: PyroModule
    guide: pyro.infer.autoguide
    neptune_run: NeptuneRun | None
    epoch: int
    every_n_epochs: int
