from pathlib import Path

import pyro
import torch
from pyro.params.param_store import ParamStoreDict
from torch.serialization import MAP_LOCATION

from src.models.initialize_model import _get_model_guide
from src.utils.local_io import get_results_dir


def _torch_load_model(source: Path, device: torch.device):
    """Load a model state dict from local path."""
    return torch.load(source, weights_only=False, map_location=device)


class _ParamStoreLoader(ParamStoreDict):
    def load(self, filename: str, map_location: MAP_LOCATION = None) -> None:
        with open(filename, "rb") as input_file:
            state = torch.load(input_file, map_location, weights_only=False)
        self.set_state(state)


def _load_parameters_local(source: Path, device: torch.device):
    params_store = _ParamStoreLoader()
    params_store.load(str(source), map_location=device)
    pyro.get_param_store().set_state(params_store.get_state())


def load_model_and_parameters(
    cfg,
    device,
    embedding_size,
):
    """Load trained model and parameters"""
    pyro.clear_param_store()

    data_base: Path = get_results_dir(cfg)
    model_dir_path = Path(cfg.model.prediction.model_dir_path)
    model_file_name = data_base / model_dir_path / "model.pth"
    pyro_params_file_name = data_base / model_dir_path / "params.pth"
    model_state_dict_with_guide = _torch_load_model(model_file_name, device)
    _load_parameters_local(pyro_params_file_name, device)
    print("end of load_model_and_parameters")
    model, guide = _get_model_guide(embedding_size, cfg)
    model.load_state_dict(model_state_dict_with_guide["model"])

    return model, guide
