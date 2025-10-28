from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyro
import torch
from omegaconf import DictConfig, OmegaConf


def get_data_dir(cfg: DictConfig) -> Path:
    """Return the base data directory from config."""
    return Path(cfg.paths.data_dir).expanduser().resolve()


def get_results_dir(cfg: DictConfig) -> Path:
    """Return the base results directory from config, ensure it exists."""
    path = Path(cfg.paths.results_dir).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_path(base: Path, relative_or_abs: str | os.PathLike[str]) -> Path:
    """Resolve a path, relative to base if not absolute."""
    p = Path(relative_or_abs)
    return p if p.is_absolute() else (base / p)


def load_csv_local(path: Path) -> pd.DataFrame:
    """Load a CSV file from local filesystem."""
    return pd.read_csv(path)


def save_npy_local(array: np.ndarray, path: Path) -> None:
    """Save a numpy array to local filesystem."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, array)


def load_yaml_local(path: Path) -> dict[str, Any]:
    """Load a YAML file into a Python dict."""
    with open(path, "r", encoding="utf-8") as f:
        return OmegaConf.to_container(
            OmegaConf.load(f), resolve=True
        )  # type: ignore[return-value]


def save_yaml_local(cfg: DictConfig, path: Path) -> None:
    """Save a Hydra/OmegaConf config to YAML locally."""
    path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config=cfg, f=str(path))


def write_bytes_local(path: Path, data: bytes) -> None:
    """Write raw bytes to a file, ensuring parent directories exist."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(data)


def copy_file_local(src: Path, dst: Path) -> None:
    """Copy a file locally creating parents."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(src, "rb") as fsrc, open(dst, "wb") as fdst:
        fdst.write(fsrc.read())


def save_model_state_local(destination: Path, state: dict[str, Any]) -> None:
    """Save a model/guide state dict locally."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, destination)


def save_pyro_params_local(destination: Path) -> None:
    """Save Pyro param store locally."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    pyro.get_param_store().save(destination)
