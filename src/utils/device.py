import torch
from omegaconf import DictConfig


def select_device(cfg: DictConfig):
    """
    Select device where to load the variables, cuda VS cpu.
    """
    device = cfg.to_cuda if cfg.to_cuda else "cpu"

    if cfg.to_cuda == "cuda" and not torch.cuda.is_available():
        print("#" * 100)
        print("CUDA is not available, using CPU")
        print("#" * 100)
        device = "cpu"

    return device
