from pathlib import Path
from typing import Optional

import pyro
import torch
from omegaconf import DictConfig

from src.models.constants import GUIDE_TYPE
from src.utils.dynamic_import import import_class_from_string


def _get_model_guide(embedding_size: tuple[int, int], cfg: DictConfig):
    """
    Select chosen architecture and instantiate.

    Parameters:
    -----------
    embedding_size: tuple[int, int]
        The size of the embedding.
    cfg: DictConfig
        The configuration object.

    Returns:
    --------
    model: PyroModule
        The model.
    guide: AutoDiagonalNormal(PhageDisplayPoissonRegressor)
        The guide.
    """
    model = import_class_from_string(cfg.model.training.distribution)(
        cfg, embedding_size
    )
    guide = GUIDE_TYPE[cfg.model.training.guide_type](
        pyro.poutine.block(model, hide=["obs"])
    )

    return model, guide


def load_or_initialize_model(
    cfg: DictConfig,
    paths: dict[str, str],
    embedding_size: tuple[int, int],
    model_checkpoint_dir: Optional[Path],
):
    """
    Loads an existing model and its parameters from the specified experiment directory
    if available, or initializes a new model, guide.

    Parameters:
    -----------
    model_checkpoint_dir (Path):
        Path to the directory where the model and parameters are stored.
    selected_model (str):
        Identifier for the model to be loaded or initialized.
    cfg (DictConfig):
        Configuration object.
    paths (dict[str, str]):
        Dictionary containing the paths where to download the model and parameters.

    Returns:
    --------
    model (PhageDisplayPoissonRegressor)
    guide (AutoDiagonalNormal(PhageDisplayPoissonRegressor) | AutoNormal(PhageDisplayPoissonRegressor))
    """
    if model_checkpoint_dir is None:
        model, guide = _get_model_guide(embedding_size, cfg)
    else:
        model_state_dict = torch.load(model_checkpoint_dir / paths["model"])
        pyro.get_param_store().load(model_checkpoint_dir / paths["parameters"])
        model, guide = _get_model_guide(embedding_size, cfg)

        model.load_state_dict(model_state_dict["model"])
        guide = model_state_dict["guide"]

    return model, guide
