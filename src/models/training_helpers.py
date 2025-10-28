# pylint: disable=R0915,R0913,R0914,R0917,undefined-variable,unused-variable
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import pyro
import torch
from neptune import Run as NeptuneRun
from omegaconf import DictConfig
from pyro.infer import Predictive
from pyro.nn import PyroModule
from scipy.stats import spearmanr
from torch.utils.data import DataLoader

from src.models.constants import PATH_DICT
from src.models.evaluation_helpers import evaluate_model
from src.models.initialize_model import load_or_initialize_model
from src.models.optimizer import initialize_optimizer
from src.models.predict_helpers import _process_predictions
from src.utils.config_utils import print_model_summary
from src.utils.evaluation_config import EvaluationConfig
from src.utils.local_io import save_model_state_local, save_pyro_params_local
from src.utils.neptune_utils import log_metrics_to_neptune, create_graph_run_from_dict


def main_loss(
    model_trace: pyro.poutine.trace_struct.Trace,
    guide_trace: pyro.poutine.trace_struct.Trace,
    annealing_factor: float,
    latents_to_anneal: list[str],
):
    """
    Calculates the ELBO loss for variational inference with KL annealing.

    Parameters:
    -----------
    model_trace (pyro.poutine.trace_struct.Trace):
        The trace object from the model, containing information about the sampled sites
        and their log probabilities.
    guide_trace (pyro.poutine.trace_struct.Trace):
        The trace object from the guide (variational distribution), containing
        information about the sampled sites and their log probabilities.
    annealing_factor (float):
        The factor by which the log probabilities of latent variables are scaled.
    latents_to_anneal (list[str]):
        List names of latent variables whose log probabilities should be scaled by the
        annealing factor.

    Returns:
    --------
    float
        The negative ELBO loss. The loss is negated because Pyro's optimizers minimize
        the objective function, and we aim to maximize the ELBO.
    """
    elbo = torch.tensor(0.0, device=model_trace.nodes["obs"]["value"].device)
    elbo_reconstruction = torch.tensor(
        0.0, device=model_trace.nodes["obs"]["value"].device
    )
    # loop through all the sample sites in the model and guide trace and
    # construct the loss; note that we scale all the log probabilities of
    # samples sites in `latents_to_anneal` by the factor `annealing_factor`
    # if no latents to anneal are provided, we use all the latent
    # variables in the model and guide and remove the obs site classic configuration)

    def treat_site(
        site: dict[str, Any],
        site_type: Optional[str],
        site_name: Optional[str],
        is_model: bool,
    ) -> torch.Tensor:
        """
        Treat a sample site in the trace for the ELBO loss.

        Parameters:
        -----------
        site: dict[str, Any]
            The site to treat.
        site_type: str
            The type of the site to treat.
        site_name: str
            The name of the site to treat.
        elbo: float
            The ELBO loss.
        is_model: bool
            Whether the site is from the model or the guide.

        Returns:
        --------
        float
            The calculated loss.
        """
        if site["type"] == site_type or site["name"] == site_name:
            # Apply annealing factor if site is in latents_to_anneal
            factor = annealing_factor if site["name"] in latents_to_anneal else 1.0
            # Calculate log probability and update ELBO
            log_prob = factor * site["fn"].log_prob(site["value"]).sum()
            return log_prob if is_model else -log_prob
        return torch.tensor(0.0)

    if len(latents_to_anneal) == 0:
        latents_to_anneal = set(model_trace.nodes.keys()) | set(
            guide_trace.nodes.keys()
        )
        latents_to_anneal.remove("obs")
    # Process model trace sites
    for site in model_trace.nodes.values():
        elbo += treat_site(site, "sample", None, is_model=True)
        elbo_reconstruction += treat_site(site, None, "obs", is_model=True)
    # Process guide trace sites
    for site in guide_trace.nodes.values():
        elbo += treat_site(site, "sample", None, is_model=False)
    return -elbo, elbo_reconstruction


def loss_and_update_fn(
    model: PyroModule,
    guide: torch.nn.Module,
    data_batch: dict[str, torch.FloatTensor],
    optimizer: torch.optim.Optimizer,
    annealing_factor: float,
    latents_to_anneal: list[str],
    training: bool = True,
    number_of_particles: int = 10,
) -> dict[str, torch.Tensor]:
    """
    Computes the loss and performs an update step for Stochastic Variational Inference
    (SVI) with KL annealing.

    Parameters:
    -----------
    model (PyroModule):
        The probabilistic model to be optimized.
    guide (pyro.infer.autoguide):
        The variational guide used to sample the latent variables.
    data_batch (dict[str, torch.FloatTensor]):
        A dictionary containing the observed data needed for the model and guide.
    optimizer (torch.optim.Optimizer):
        The optimizer used to update the model parameters.
    annealing_factor (float):
        The factor to scale the log probabilities of specified latent variables.
    latents_to_anneal (list[str]):
        A list of latent variables whose log probabilities should be scaled by the
        annealing factor.

    Returns:
    --------
    dict[str, torch.Tensor]
        Dictionary containing the computed ELBO loss metrics for the given data.
    """
    # Initialize lists to store loss tensors from each particle
    elbo_particles = []
    recon_particles = []

    for _particle_idx in range(number_of_particles):  # needs to be in the config
        guide_trace = pyro.poutine.trace(guide).get_trace(data_batch)
        model_trace = pyro.poutine.trace(
            pyro.poutine.replay(model, trace=guide_trace)
        ).get_trace(data_batch)
        loss_val_particle, elbo_reconstruction_particle = main_loss(
            model_trace,
            guide_trace,
            annealing_factor,
            latents_to_anneal,
        )

        # Add loss tensors to lists
        elbo_particles.append(loss_val_particle)
        recon_particles.append(elbo_reconstruction_particle)
        del guide_trace, model_trace

    elbo_loss_total = torch.stack(elbo_particles).sum() / number_of_particles
    recon_loss_total = torch.stack(recon_particles).sum() / number_of_particles
    kl_loss_total = elbo_loss_total + recon_loss_total

    metrics = {
        "ELBOloss": elbo_loss_total,
        "ReconstructionLoss": -recon_loss_total,
        "KLloss": kl_loss_total,
    }

    if training:
        # Backpropagation is done on the final averaged loss tensor
        metrics["ELBOloss"].backward()
        optimizer.step()
        optimizer.zero_grad()

    # Detach metrics from computational graph for logging
    final_metrics = {}
    for metric_name, metric_value in metrics.items():
        # detach from cuda and divide by total sequences in experiment
        # to get per sequence loss
        final_metrics[metric_name] = (
            metric_value.detach().cpu() / data_batch["total_sequences_in_experiment"]
        )
    return final_metrics


def _init_epoch_metrics(include_test: bool) -> dict[str, dict[str, float]]:
    """Initialize per-dataset metrics containers for the epoch.

    This ensures a consistent structure and reduces duplication in the
    training/evaluation loops.
    """
    metrics: dict[str, dict[str, float]] = {
        "train": {"ELBOloss": 0.0, "ReconstructionLoss": 0.0, "KLloss": 0.0},
        "valid": {"ELBOloss": 0.0, "ReconstructionLoss": 0.0, "KLloss": 0.0},
    }
    if include_test:
        metrics["test"] = {"KD_correlation": 0.0}
    return metrics


def _process_train_valid_batch(
    model: PyroModule,
    guide: torch.nn.Module,
    data_batch: dict[str, torch.Tensor],
    is_training: bool,
    metrics: dict[str, dict[str, float]],
    optimizer: torch.optim.Optimizer,
    annealing_factor: float,
    latents_to_anneal: list[str],
    number_of_particles: int,
) -> dict[str, dict[str, float]]:
    """Process a single train/valid batch and update metrics in-place."""
    metric_dict = loss_and_update_fn(
        model,
        guide,
        data_batch,
        optimizer,
        annealing_factor,
        latents_to_anneal,
        training=is_training,
        number_of_particles=number_of_particles,
    )
    dataset_type = "train" if is_training else "valid"
    for metric_name, metric_value in metric_dict.items():
        metrics[dataset_type][metric_name] += metric_value.item()
    return metrics


def _process_test_batch(
    model: PyroModule,
    guide: torch.nn.Module,
    data_batch: dict[str, torch.Tensor],
    metrics: dict[str, dict[str, float]],
    kd_values: Optional[list[float]],
    num_prediction_particles: int,
) -> dict[str, dict[str, float]]:
    """Process a single test batch and update KD correlation metric in-place."""
    with torch.no_grad():
        predictive_object = Predictive(
            model, guide=guide, num_samples=num_prediction_particles
        )
        data_for_inference = data_batch.copy()
        data_for_inference["selected_count"] = None
        data_for_inference["selected_frequency"] = None
        data_for_inference["selectivity"] = None
        posterior_samples = predictive_object(data_for_inference)
        (batch_mean, _, _, _, _, _, _, _, _) = _process_predictions(posterior_samples)

    prediction_df = pd.DataFrame(
        {"predicted_KD": batch_mean, "sequence": data_for_inference["sequence"]}
    ).set_index("sequence")

    true_df = pd.DataFrame(
        {"sequence": kd_values["sequence"], "true_KD": kd_values["KD_values"]}
    ).set_index("sequence")
    merged_df = true_df.join(prediction_df, how="inner", on="sequence")

    metrics["test"]["KD_correlation"] += spearmanr(
        merged_df["true_KD"], merged_df["predicted_KD"]
    )[0]
    return metrics


def train_bayesian_elbo_one_epoch(
    model: PyroModule,
    guide: pyro.infer.autoguide,
    optimizer: torch.optim.AdamW,
    scheduler: torch.optim.lr_scheduler.SequentialLR,
    train_dataloader: DataLoader,
    valid_dataloader: DataLoader,
    test_dataloader: DataLoader,
    epoch: int,
    cfg: DictConfig,
    neptune_run: NeptuneRun,
    kd_values: Optional[list[float]],
    is_valid: bool = False,
) -> float:
    """
    Performs one epoch of training and evaluation for a Bayesian model using ELBO
    optimization. The training employs KL annealing for regularization, which adjusts
    the contribution of the KL divergence term in the ELBO loss over epochs.

    Parameters:

    -----------
    training_loop_dict (dict[str, Any]):
        Dictionary containing training components and other relevant settings.
    model (PyroModule):
        The probabilistic model which approximates the likelihood of the data.
    guide (pyro.infer.autoguide):
        The guide (approximate posterior) used for variational inference.
    train_dataloader (DataLoader):
        DataLoader for the training dataset.
    valid_dataloader (DataLoader):
        DataLoader for the validation dataset.
    epoch (int):
        The current epoch number.
    cfg [DictConfig]:
        A configuration dictionary.
    kd_values (Optional[list[float]]):
        The KD values for the test dataset.

    Returns:

    --------
    float
        The total loss for the current epoch.

    Notes:
    ------
    - KL Annealing: Gradually increases the weight of the KL divergence term in the ELBO
      loss to stabilize training.
    - The learning rate scheduler in `training_loop_dict["scheduler"]` is stepped at the
      end of each epoch to adjust the learning rate.
    """
    latents_to_anneal, annealing_factor = handle_annealing(epoch, cfg)

    # Initialize metrics for training, validation, and optionally test
    include_test = bool(test_dataloader and kd_values is not None)
    metrics = _init_epoch_metrics(include_test)

    # Training loop
    model.init_sum_list_to_zero()
    torch.cuda.empty_cache()

    batch_count = 0
    for data_batch in train_dataloader:
        metrics = _process_train_valid_batch(
            model=model,
            guide=guide,
            data_batch=data_batch,
            is_training=True,
            metrics=metrics,
            optimizer=optimizer,
            annealing_factor=annealing_factor,
            latents_to_anneal=latents_to_anneal,
            number_of_particles=cfg.model.training.number_of_particles,
        )
        batch_count += 1
        experiment_name = data_batch["experiment_name"][0]

    scheduler.step()

    # Update intermediate sum cache at the end of epoch
    with torch.no_grad():
        model.update_intermediate_sum_cache(experiment_name)

        # Validation loop
        if is_valid:
            for data_batch in valid_dataloader:
                metrics = _process_train_valid_batch(
                    model=model,
                    guide=guide,
                    data_batch=data_batch,
                    is_training=False,
                    metrics=metrics,
                    optimizer=optimizer,
                    annealing_factor=annealing_factor,
                    latents_to_anneal=latents_to_anneal,
                    number_of_particles=cfg.model.training.number_of_particles,
                )
            if test_dataloader and kd_values is not None:
                for data_batch in test_dataloader:
                    _process_test_batch(
                        model=model,
                        guide=guide,
                        data_batch=data_batch,
                        metrics=metrics,
                        kd_values=kd_values,
                        num_prediction_particles=(
                            cfg.model.prediction.number_of_particles
                        ),
                    )

    # Import locally to avoid circular import
    evaluation_config = EvaluationConfig(
        model=model,
        guide=guide,
        neptune_run=neptune_run,
        epoch=epoch,
        every_n_epochs=cfg.model.training.every_n_epochs,
    )
    keep_preds = evaluate_model(evaluation_config, train_dataloader, valid_dataloader)

    print(
        "Training set ground truth vs predicted counts (observations): mean & std\n",
        f"\tmean: {np.mean(np.vstack(keep_preds['train']))}",
        f"std: {np.std(np.vstack(keep_preds['train']))}",
    )
    print(
        "Validation set ground truth vs predicted counts (observations): mean & std\n",
        f"\tmean: {np.mean(np.vstack(keep_preds['valid']))}",
        f"std: {np.std(np.vstack(keep_preds['valid']))}",
    )

    return metrics


def handle_annealing(epoch: int, cfg: DictConfig) -> tuple[list[str], float]:
    """This function will return the latents to anneal and the annealing factor.
    The annealing factor is a float between 0 and 1 that will be used to scale the
    log probabilities of the latents to anneal in order to focus on the reconstruction
    loss when the annealing factor is high (the annealing factor is dependent on the
    epoch and the kl_annealing_type).

    Parameters:
    -----------
    epoch: int
        The current epoch number.
    cfg: DictConfig
        The configuration object.

    Returns:
    --------
    tuple[list[str], float]
        The latents to anneal and the annealing factor.
    """
    latents_to_anneal = cfg.model.training.latents_to_anneal
    if latents_to_anneal is None:
        latents_to_anneal = []
    kl_annealing, kl_annealing_type = (
        cfg.model.training.kl_annealing,
        cfg.model.training.kl_annealing_type,
    )
    match kl_annealing_type:
        case None:
            annealing_factor = 1.0
        case "linear":
            annealing_factor = (epoch + 1) / kl_annealing
        case "exponential":
            annealing_factor = (
                1 - np.exp(-(epoch + 1) / kl_annealing) if kl_annealing > 1 else 1.0
            )
        case "cyclic":
            annealing_factor = (epoch + 1) % (2 * kl_annealing) / kl_annealing
        case _:
            raise ValueError(f"Invalid KL annealing type: {kl_annealing}")
    annealing_factor = min(annealing_factor, 1)
    return latents_to_anneal, annealing_factor


def _setup_model_and_optimizer(
    cfg: DictConfig,
    embedding_size: tuple[int, int],
    device: torch.device,
    train_dataloader: DataLoader,
    model_checkpoint_dir: Optional[Path],
):
    """Setup model, guide, and optimizer for training.

    Returns:
        tuple: (model, guide, scheduler, optimizer)
    """
    model, guide = load_or_initialize_model(
        cfg,
        PATH_DICT,
        embedding_size,
        model_checkpoint_dir,
    )
    # Move model and guide to device BEFORE initializing optimizer
    model.to(device)
    guide.to(device)
    scheduler, optimizer = initialize_optimizer(
        dataloader=train_dataloader,
        model=model,
        guide=guide,
        cfg=cfg,
    )
    return model, guide, scheduler, optimizer


def _save_best_model(
    model: PyroModule,
    guide: pyro.infer.autoguide,
    experiment_dir: Path,
) -> dict[str, torch.Tensor]:
    """Save the best model based on selected model type."""
    save_model_state_local(
        experiment_dir / PATH_DICT["model"],
        {"model": model.state_dict(), "guide": guide.state_dict()},
    )
    save_pyro_params_local(experiment_dir / PATH_DICT["parameters"])


def _log_first_epoch_info(model, neptune_run: NeptuneRun) -> None:
    """Log model information for the first epoch."""
    neptune_run["model/TrainableParameters"] = count_parameters(
        model, trainable_only=False
    )
    neptune_run["model/Summary"] = print_model_summary(model)
    assert count_parameters(model, trainable_only=False) > 0


def _print_epoch_results(
    metrics: dict,
    best_epoch: int,
    best_loss: float,
    epoch_duration: float,
) -> None:
    """Print results for the current epoch."""
    print(
        f"Average training elbo loss: "
        f"{-1 * metrics['train']['ELBOloss']: .3f}\n"
        f"Average training reconstruction loss: "
        f"{-1 * metrics['train']['ReconstructionLoss']: .3f}\n"
        f"Average training kl loss: "
        f"{-1 * metrics['train']['KLloss']: .3f}\n"
        f"Best epoch: [{best_epoch}], best elbo loss: [{-1 * best_loss: .3f}]\n"
        f"epoch duration: {epoch_duration: .3f}s"
    )


def train_bnn(
    train_dataloader: DataLoader,
    valid_dataloader: DataLoader,
    test_dataloader: DataLoader,
    cfg: DictConfig,
    experiment_dir: Path,
    model_checkpoint_dir: Optional[Path],
    neptune_run: NeptuneRun,
    device: torch.device,
    embedding_size: tuple[int, int],
    kd_values: Optional[list[float]],
) -> None:
    """
    Training loop for a bayesian model with ELBO loss.

    Parameters:
    -----------
    train_dataloader (DataLoader):
        DataLoader for the training dataset.
    valid_dataloader (DataLoader):
        DataLoader for the validation dataset.
    cfg [DictConfig]:
        Configuration object.
    experiment_dir (Path):
        path for the experiment.
    model_checkpoint_dir (Path):
        Path to the model checkpoint
    neptune_run (NeptuneRun):
        Neptune object to load metrics.

    Returns:
    --------
    dict[str, torch.Tensor]
    """
    pyro.clear_param_store()

    selected_model = cfg.model.type
    print(f"---- SELECTED MODEL: {selected_model} ----")
    # Setup model, guide, and optimizer
    model, guide, scheduler, optimizer = _setup_model_and_optimizer(
        cfg,
        embedding_size,
        device,
        train_dataloader,
        model_checkpoint_dir,
    )

    best_loss = np.inf
    best_epoch = 0

    print("\n######################################\nStarting training loop...\n")

    for epoch in range(cfg.model.training.epochs):
        epoch_start_time = time.time()
        print(f"\n------- EPOCH {epoch + 1} -------")

        # Train one epoch
        metrics = train_bayesian_elbo_one_epoch(
            model,
            guide,
            optimizer,
            scheduler,
            train_dataloader,
            valid_dataloader,
            test_dataloader,
            epoch,
            cfg,
            neptune_run,
            is_valid=True,
            kd_values=kd_values,
        )
        # Save model if it's the best so far
        with torch.no_grad():
            if (
                metrics[cfg.model.training.best_model_decision_set]["ELBOloss"]
                < best_loss
            ):
                _save_best_model(
                    model,
                    guide,
                    experiment_dir,
                )
                best_loss = metrics[cfg.model.training.best_model_decision_set][
                    "ELBOloss"
                ]
                best_epoch = epoch

        dataset_types = ["train", "valid"]

        # If Neptune run is a dict, we need to initialize the epoch_duration list.
        if isinstance(neptune_run, dict) and "epoch_duration" not in neptune_run:
            neptune_run["epoch_duration"] = []

        # Log metrics to Neptune
        for dataset_type in dataset_types:
            if neptune_run is not None:
                log_metrics_to_neptune(metrics[dataset_type], neptune_run, dataset_type)

        # Log first epoch information
        epoch_duration = time.time() - epoch_start_time
        if epoch == 0 and cfg.use_neptune and neptune_run is not None:
            _log_first_epoch_info(model, neptune_run)

        # Print epoch results
        _print_epoch_results(metrics, best_epoch, best_loss, epoch_duration)

        if neptune_run is not None:
            neptune_run["epoch_duration"].append(epoch_duration)
    if isinstance(neptune_run, dict):
        print("Creating graphs from Neptune run")
        create_graph_run_from_dict(neptune_run)
    print(
        f"\nEnd of training loop epoch {epoch + 1}.\n",
        "######################################\n",
    )


def count_parameters(model: torch.nn.Module, trainable_only: bool = True) -> int:
    """
    Count the number of parameters in a PyTorch model, including Pyro models.

    Parameters:
        model: PyTorch model
        trainable_only: If True, count only trainable parameters

    Returns:
        Number of parameters
    """
    # First try standard PyTorch parameter counting
    if trainable_only:
        standard_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        standard_params = sum(p.numel() for p in model.parameters())

    # For Pyro models, estimate from layer structure
    total_pyro_params = 0
    for name, value in pyro.get_param_store().items():
        print(f"Pyro param: {name}, shape: {value.shape}")
        total_pyro_params += value.numel()
    return total_pyro_params + standard_params
