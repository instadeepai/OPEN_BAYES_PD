import pyro
import torch.optim
from omegaconf import DictConfig
from pyro.nn import PyroModule
from torch.optim.lr_scheduler import ExponentialLR, LinearLR, SequentialLR

from src.data_processing.dataloader import PhageDisplayDataLoader
from src.models.constants import optimizer_dict


def initialize_optimizer(
    dataloader: PhageDisplayDataLoader,
    model: PyroModule,
    guide: pyro.infer.autoguide,
    cfg: DictConfig,
) -> tuple[SequentialLR, torch.optim.Optimizer]:
    """
    Initializes the optimizer and scheduler with the right model parameters and
    configuration hyperparameter, in a Pyro model context with ELBO loss.

    Parameters:
    -----------
    dataloader [PhageDisplayDataLoader]:
        Custom object for loading data batches.
    model [PyroModule]:
        The probabilistic model.
    guide [pyro.infer.autoguide]:
        A guide function for variational inference.
    cfg [DictConfig]:
        A configuration dictionary.

    Returns:
    --------
    tuple[torch.optim.lr_scheduler.SequentialLR, torch.optim.Optimizer]:
            learning rate scheduler and optimizer.
    """
    optimizer_parameters = {
        "lr": cfg.model.training.learning_rate,
        "weight_decay": cfg.model.training.weight_decay,
    }
    initialization_batch = next(iter(dataloader))
    # Record the sampling and parameter operations in the guide.
    guide_trace = pyro.poutine.trace(guide).get_trace(initialization_batch)
    # Use the guide's trace to impose its random variable values on the model.
    # ensuring consistency with the guide's random variables.
    model_trace = pyro.poutine.trace(
        pyro.poutine.replay(model, trace=guide_trace)
    ).get_trace(initialization_batch)

    model_params = {
        node["value"].unconstrained()
        for node in model_trace.nodes.values()
        if node["type"] == "param"
    }
    guide_params = {
        node["value"].unconstrained()
        for node in guide_trace.nodes.values()
        if node["type"] == "param"
    }

    # merge parameters: optimize both model and guide to minimize ELBO during SVI
    merged_params = model_params | guide_params
    optimizer = optimizer_dict[cfg.model.training.optimizer](
        merged_params, **optimizer_parameters
    )
    scheduler_initial = LinearLR(
        optimizer,
        start_factor=1.0 / cfg.model.training.end_of_warmup,
        total_iters=cfg.model.training.end_of_warmup,
    )
    scheduler_final = ExponentialLR(optimizer, gamma=cfg.model.training.gamma_scheduler)
    scheduler = SequentialLR(
        optimizer,
        schedulers=[scheduler_initial, scheduler_final],
        milestones=[cfg.model.training.end_of_warmup],
    )
    return scheduler, optimizer
