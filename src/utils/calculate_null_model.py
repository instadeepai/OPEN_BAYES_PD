import numpy as np
import pyro
import torch
from pyro.infer import Predictive
from scipy.stats import spearmanr

from src.models.constants import NUM_SAMPLES
from src.utils.dynamic_import import import_class_from_string


def calculate_null_model_correlation(
    dataloader,
    experiment,
    cfg,
    seed: int = 42,
) -> float:
    """
    Calculates the correlation between the null model's predicted abundance
    using the appropriate null model class (Poisson or Multinomial) and the
    observed counts.

    Parameters:
    ----------
    dataloader: A PhageDisplayDataLoader object that yields dictionaries containing:
        'selected_count': The observed counts after selection.
        'selected_total': Total selected count.
    experiment: str
        The experiment name to filter data.
    cfg: DictConfig
        The configuration object.
    seed: int
        Random seed for reproducibility.
    number_of_particles: int
        Number of samples to generate for each sequence.

    Returns:
    -------
    float: The Spearman correlation coefficient between the null model's predicted
        counts and the observed counts.
    """
    # Set random seeds for reproducibility
    torch.manual_seed(seed)
    pyro.set_rng_seed(seed)

    # Initialize the appropriate null model based on configuration
    null_model = import_class_from_string(cfg.model.training.distribution)(
        cfg, embedding_size=(1, 1), is_null_model=True
    )

    # Collect all batches with all necessary data
    all_dicts = {
        "selected_count_ground_truth": [],
        "initial_count": [],
        "initial_total": [],
        "selected_total": [],
        "total_sequences_in_experiment": [],
        "experiment_name": [],
    }

    for batch in dataloader:
        if experiment in batch["experiment_name"][0]:
            # Create dummy embeddings since the null model doesn't use them
            batch["embeddings"] = torch.ones((batch["initial_count"].shape[0], 1))
            all_dicts["selected_count_ground_truth"].append(batch["selected_count"])
            all_dicts["initial_count"].append(batch["initial_count"])
            all_dicts["initial_total"].append(batch["initial_total"])
            all_dicts["selected_total"].append(batch["selected_total"])
            all_dicts["experiment_name"].append(batch["experiment_name"])
            all_dicts["total_sequences_in_experiment"].append(
                batch["total_sequences_in_experiment"]
            )

    # Concatenate all batches
    all_dicts_tensors = {
        k: torch.cat(all_dicts[k], dim=0).squeeze()
        for k in [
            "initial_count",
            "initial_total",
            "selected_total",
            "selected_count_ground_truth",
        ]
    }
    all_dicts_tensors["embeddings"] = torch.ones(
        (all_dicts_tensors["initial_count"].shape[0], 1)
    )
    all_dicts_tensors["selected_count"] = None
    all_dicts_tensors["total_sequences_in_experiment"] = all_dicts[
        "total_sequences_in_experiment"
    ][0]
    all_dicts_tensors["experiment_name"] = ["null_model"]
    # Generate predictions using the null model
    with torch.no_grad():
        predictive = Predictive(
            model=null_model,
            num_samples=NUM_SAMPLES,
        )
        predictions = predictive(all_dicts_tensors)
        posterior_predictions = predictions["obs"].detach().cpu().numpy()

    # Convert predictions to numpy array and calculate mean
    posterior_predictions = np.stack(posterior_predictions)
    mean_predictions = np.mean(posterior_predictions.squeeze(), axis=0)
    observed_counts = (
        all_dicts_tensors["selected_count_ground_truth"].detach().cpu().numpy()
    )

    # Calculate correlation
    correlation = spearmanr(mean_predictions, observed_counts)[0]

    return correlation
