"""
Shape Suffixes Documentation:

S: sequences (number of sequences in the experiment)
N: number of samples (for posterior sampling)

Example:
- posterior_samples_SN: posterior samples with sequences and number of samples
"""


from functools import lru_cache
from typing import Any, Dict, NamedTuple

import numpy as np
import torch
from pyro.infer import Predictive
from scipy.stats import spearmanr
from torch.utils.data import DataLoader

from src.models.constants import NUM_SAMPLES
from src.utils.evaluation_config import EvaluationConfig
from src.utils.neptune_utils import log_metrics_to_neptune
from src.utils.plots import scatter_plot_predictions


class ExperimentMetrics(NamedTuple):
    """Container for experiment metrics data."""

    mean_binding_S: np.ndarray
    mean_negative_binding_S: np.ndarray
    select_S: np.ndarray
    counts_S: np.ndarray
    mean_pred_S: np.ndarray


class PlotData(NamedTuple):
    """Container for plot data."""

    binding_probs_raw_SN: np.ndarray
    negative_binding_probs_SN: np.ndarray
    preds_SN: np.ndarray
    counts_S: np.ndarray
    select_S: np.ndarray
    initial_counts_S: np.ndarray


class PlotGenerationData(NamedTuple):
    """Container for plot generation data."""

    name: str
    data: Dict[str, Any]
    binding_probs_raw_SN: np.ndarray
    select_S: np.ndarray
    counts_S: np.ndarray
    initial_counts_S: np.ndarray
    config: EvaluationConfig
    dataset_type: str


def _evaluate_model_on_dataset(
    config: EvaluationConfig,
    dataloader: DataLoader,
    dataset_type: str,
) -> list[np.ndarray]:
    """Evaluate after aggregating all mini-batches of each experiment.

    Parameters:
    -----------
    config (EvaluationConfig):
        Configuration object containing model, guide, and logging parameters.
    dataloader (DataLoader):
        DataLoader for the dataset.
    dataset_type (str):
        The type of dataset to evaluate.

    Returns:
    --------
    list[np.ndarray]
        A list of absolute differences between predicted and observed counts.
    """
    with torch.no_grad():
        # Use fewer samples for faster evaluation
        predictive = Predictive(
            config.model, guide=config.guide, num_samples=NUM_SAMPLES
        )

        # Initialize data structures
        experiment_data = {
            "experiments": {},
            "keep_diffs": [],
            "pool_list": {
                "binding_list": [],
                "select_list": [],
                "preds_list": [],
                "counts_list": [],
            },
        }

        # Process batches
        batch_count = 0
        for data_batch in dataloader:
            _process_evaluation_batch(data_batch, predictive, experiment_data)
            batch_count += 1

        # Process experiments and generate metrics
        _process_experiments(experiment_data, config, dataset_type)

    return experiment_data["keep_diffs"]


def _process_evaluation_batch(
    data_batch: dict[str, Any], predictive: Predictive, experiment_data: dict[str, Any]
) -> None:
    """Process a single batch and update experiment data.

    Parameters:
    -----------
    data_batch: Batch of data containing sequences and their metadata
    predictive: Predictive object for generating posterior samples
    experiment_data: Dictionary to store aggregated experiment data
    """
    print(f"batch: {data_batch['embeddings'].shape}")
    name = data_batch["experiment_name"][0]

    # Initialize experiment if not exists
    if name not in experiment_data["experiments"]:
        experiment_data["experiments"][name] = {
            "binding_probs_SN": [],
            "negative_binding_probs_SN": [],
            "selectivities_S": [],
            "predictions_SN": [],
            "counts_S": [],
            "initial_counts_S": [],
        }

    # Prepare batch for inference
    batch_inf = data_batch.copy()
    batch_inf["selected_count"] = None
    batch_inf["selectivity"] = None

    # Generate posterior samples
    posterior_samples_SN = predictive(batch_inf)

    # Extract data efficiently
    batch_data = {
        "binding_probs_raw_SN": (
            posterior_samples_SN["binding_probability"].detach().cpu().numpy()
        ),
        "negative_binding_probs_SN": (
            posterior_samples_SN["negative_binding_probability"].detach().cpu().numpy()
        ),
        "selectivities_S": data_batch["selectivity"].detach().cpu().numpy().flatten(),
        "predictions_SN": posterior_samples_SN["obs"].detach().cpu().numpy().T,
        "counts_S": data_batch["selected_count"].detach().cpu().numpy().flatten(),
        "initial_counts_S": data_batch["initial_count"]
        .detach()
        .cpu()
        .numpy()
        .flatten(),
    }

    # Update experiment data
    exp = experiment_data["experiments"][name]
    exp["binding_probs_SN"].append(batch_data["binding_probs_raw_SN"])
    exp["negative_binding_probs_SN"].append(batch_data["negative_binding_probs_SN"])
    exp["selectivities_S"].append(batch_data["selectivities_S"])
    exp["predictions_SN"].append(batch_data["predictions_SN"])
    exp["counts_S"].append(batch_data["counts_S"])
    exp["initial_counts_S"].append(batch_data["initial_counts_S"])

    # Calculate differences
    experiment_data["keep_diffs"].append(
        np.abs(
            batch_data["predictions_SN"].squeeze()
            - batch_data["counts_S"].reshape(-1, 1)
        )
    )

    # Clear batch data and samples to free memory
    del batch_data, posterior_samples_SN, batch_inf


@lru_cache(maxsize=1000)
def _find_concat_axis(shapes: tuple[tuple[int, ...], ...]) -> int:
    """Find axes where shapes vary across binding probability arrays.

    Parameters
    ----------
    shapes : tuple[tuple[int, ...], ...]
        Tuple of array shapes to analyze

    Returns
    -------
    int
        Axis where shapes vary, 1 if no variation
    """
    # Skip empty lists
    if not shapes:
        return []

    # Find the axis where shapes vary
    varying_axes = []
    reference_shape = shapes[0]

    # For each axis in the reference shape
    for axis in range(len(reference_shape)):
        # Check if any other shape has a different size on this axis
        if any(s[axis] != reference_shape[axis] for s in shapes):
            varying_axes.append(axis)

    concat_axis = varying_axes[0] if varying_axes else 1

    return concat_axis


def _process_binding_probabilities(
    experiment_data: Dict[str, Any]
) -> Dict[str, np.ndarray]:
    """
    Process binding probabilities for an experiment.

    Args:
        experiment_data: Dictionary containing experiment data
        with binding probabilities.

    Returns:
        Dictionary with binding probabilities data (binding_probs_raw,
        negative_binding_raw, mean_binding, mean_negative_binding).
    """
    shapes = tuple(bp.shape for bp in experiment_data["binding_probs_SN"])
    concat_axis = _find_concat_axis(shapes)

    binding_probs_raw_SN = np.concatenate(
        experiment_data["binding_probs_SN"], axis=concat_axis
    )
    negative_binding_raw_SN = np.concatenate(
        experiment_data["negative_binding_probs_SN"], axis=concat_axis
    )

    mean_binding_S = np.mean(binding_probs_raw_SN, axis=0).squeeze()
    mean_negative_binding_S = np.mean(negative_binding_raw_SN, axis=0).squeeze()

    return {
        "binding_probs_raw_SN": binding_probs_raw_SN,
        "negative_binding_raw_SN": negative_binding_raw_SN,
        "mean_binding_S": mean_binding_S,
        "mean_negative_binding_S": mean_negative_binding_S,
    }


def _prepare_correlation_data(experiment_data: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """
    Prepare data needed for correlation calculations.

    Args:
        experiment_data: Dictionary containing experiment data
        with predictions and counts.

    Returns:
        Dictionary with correlation data (select, counts, initial_counts, mean_pred).
    """
    predictions_SN = np.vstack(experiment_data["predictions_SN"])
    selectivities_S = np.concatenate(experiment_data["selectivities_S"])
    observed_counts_S = np.concatenate(experiment_data["counts_S"])
    initial_counts_S = np.concatenate(experiment_data["initial_counts_S"])
    mean_predictions_S = np.mean(predictions_SN.squeeze(), axis=1)

    return {
        "select_S": selectivities_S,
        "counts_S": observed_counts_S,
        "initial_counts_S": initial_counts_S,
        "mean_pred_S": mean_predictions_S,
    }


def _calculate_experiment_metrics(
    name: str, metrics_data: ExperimentMetrics
) -> Dict[str, float]:
    """
    Calculate metrics for a single experiment.

    Args:
        name: Experiment name.
        metrics_data: Container with all metrics data.

    Returns:
        Dictionary of calculated metrics.
    """
    return {
        f"{name}-BindingProbSpearmanR": spearmanr(
            metrics_data.mean_negative_binding_S, metrics_data.select_S
        )[0],
        f"{name}-BindingProbSequenceMean": float(np.mean(metrics_data.mean_binding_S)),
        f"{name}-NegativeBindingProbSequenceMean": float(
            np.mean(metrics_data.mean_negative_binding_S)
        ),
        f"{name}-NegativeBindingProbSpearmanR": spearmanr(
            metrics_data.mean_negative_binding_S, metrics_data.select_S
        )[0],
        f"{name}-CountSpearmanR": spearmanr(
            metrics_data.mean_pred_S, metrics_data.counts_S
        )[0],
        f"{name}-BindingProbabilityWithout0SelectivitySpearmanR": spearmanr(
            metrics_data.mean_binding_S[metrics_data.select_S > 0],
            metrics_data.select_S[metrics_data.select_S > 0],
        )[0],
    }


def _generate_plots_if_needed(
    config: EvaluationConfig,
    dataset_type: str,
    name: str,
    plot_data: PlotData,
) -> None:
    """
    Generate plots if the current epoch matches the plotting schedule.

    Args:
        config: Configuration object.
        dataset_type: Type of dataset.
        name: Experiment name.
        plot_data: Container with all plot data.
    """
    if config.epoch % config.every_n_epochs == 0:
        scatter_plot_predictions(
            {
                "predictions_SN": plot_data.preds_SN,
                "counts_S": plot_data.counts_S,
                "binding_probs_SN": plot_data.binding_probs_raw_SN,
                "selectivities_S": plot_data.select_S,
                "initial_counts_S": plot_data.initial_counts_S,
                "negative_binding_probs_SN": plot_data.negative_binding_probs_SN,
            },
            config.neptune_run,
            dataset_type,
            name,
            config.epoch,
        )


def _process_single_experiment(
    name: str,
    experiment_data_dict: Dict[str, Any],
    experiment_data: Dict[str, Any],
    config: EvaluationConfig,
    dataset_type: str,
) -> None:
    """
    Process a single experiment and generate its metrics.

    Args:
        name: Experiment name.
        experiment_data_dict: Dictionary containing experiment data.
        experiment_data: Dictionary containing all experiment data.
        config: Configuration object.
        dataset_type: Type of dataset.
    """
    # Process binding probabilities and correlation data
    binding_data = _process_binding_probabilities(experiment_data_dict)
    correlation_data = _prepare_correlation_data(experiment_data_dict)

    # Create metrics data container
    metrics_data = ExperimentMetrics(
        mean_binding_S=binding_data["mean_binding_S"],
        mean_negative_binding_S=binding_data["mean_negative_binding_S"],
        select_S=correlation_data["select_S"],
        counts_S=correlation_data["counts_S"],
        mean_pred_S=correlation_data["mean_pred_S"],
    )

    # Accumulate data for pooled correlation
    pool_list = experiment_data["pool_list"]
    pool_list["binding_list"].append(metrics_data.mean_binding_S)
    pool_list["select_list"].append(metrics_data.select_S)
    pool_list["preds_list"].append(metrics_data.mean_pred_S)
    pool_list["counts_list"].append(metrics_data.counts_S)

    # Calculate and log metrics
    metrics = _calculate_experiment_metrics(name, metrics_data)
    log_metrics_to_neptune(metrics, config.neptune_run, dataset_type)

    # Generate plots if needed
    # if config.epoch % config.every_n_epochs == 0:
    #    plot_data = PlotData(
    #        binding_probs_raw_SN=binding_data["binding_probs_raw_SN"],
    #        negative_binding_probs_SN=binding_data["negative_binding_raw_SN"],
    #        preds_SN=np.vstack(experiment_data_dict["predictions_SN"]),
    #        counts_S=metrics_data.counts_S,
    #        select_S=metrics_data.select_S,
    #        initial_counts_S=correlation_data["initial_counts_S"],
    #    )
    # _generate_plots_if_needed(config, dataset_type, name, plot_data)

    # Clean up large arrays to free memory
    del binding_data


def _process_experiments(
    experiment_data: dict[str, Any], config: EvaluationConfig, dataset_type: str
) -> None:
    """Process all experiments and generate metrics."""
    for name, data in experiment_data["experiments"].items():
        _process_single_experiment(name, data, experiment_data, config, dataset_type)

    # Process pooled correlation
    _process_pooled_correlation(experiment_data, config, dataset_type)


def _process_pooled_correlation(
    experiment_data: dict[str, Any], config: EvaluationConfig, dataset_type: str
) -> None:
    """Process pooled correlation across all experiments."""
    for key in experiment_data["pool_list"]:
        experiment_data["pool_list"][key] = np.concatenate(
            experiment_data["pool_list"][key]
        )

    log_metrics_to_neptune(
        {
            "PooledBindingProbSpearmanR": spearmanr(
                experiment_data["pool_list"]["binding_list"],
                experiment_data["pool_list"]["select_list"],
            )[0],
            "PooledCountSpearmanR": spearmanr(
                experiment_data["pool_list"]["preds_list"],
                experiment_data["pool_list"]["counts_list"],
            )[0],
        },
        config.neptune_run,
        dataset_type,
    )


def evaluate_model(
    config: EvaluationConfig,
    train_dataloader: DataLoader,
    valid_dataloader: DataLoader,
) -> dict[str, list[float]]:
    """
    Function to evaluate the bayesian model on train and validation set.

    Parameters:
    -----------
    model (PyroModule):
        The probabilistic model which approximates the likelihood of the data.
    guide (pyro.infer.autoguide):
        The guide (approximate posterior) used for variational inference.
    train_dataloader (DataLoader):
        DataLoader for the training dataset.
    valid_dataloader (DataLoader):
        DataLoader for the validation dataset.
    neptune_run (NeptuneRun):
        Neptune object to load metrics.
    epoch (int):
        Current epoch number.
    every_n_epochs (int):
        Frequency of plot generation.

    Returns:
    --------
    dict[str, list[float]]
        A dictionary containing the predictions for the training and validation sets.
    """
    print("=== IN EVALUATE MODEL FUNCTION ===")

    # Create configuration object
    keep_preds = {}
    dataloader_dict = {
        "train": train_dataloader,
        "valid": valid_dataloader,
    }

    for dataset_type in ["train", "valid"]:
        torch.cuda.empty_cache()

        keep_preds[dataset_type] = _evaluate_model_on_dataset(
            config,
            dataloader_dict[dataset_type],
            dataset_type,
        )

    return keep_preds
