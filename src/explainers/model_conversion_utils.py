# pylint: disable=R0913, E1129

import torch
from omegaconf import DictConfig
from pyro import poutine
from pyro.infer.autoguide import AutoDiagonalNormal
from pyro.nn import PyroModule

from src.models.core.bayesian_cnn import CNNConfig, DeterministicBayesianCNNCounterpart


def create_cnn_config(cfg: DictConfig) -> CNNConfig:
    """Create CNN configuration with training parameters."""
    training_params = {
        "dropout_value": float(cfg.model.training.dropout_value),
        "activation_name": str(cfg.model.training.activation_name),
        "prior_scale": float(cfg.model.training.prior_scale),
    }
    cnn_config_dict = dict(cfg.model.fitness_predictor_cnn)
    cnn_config_dict.update(training_params)
    return CNNConfig(**cnn_config_dict)


def sample_weights_from_guide(
    guide_pyro: PyroModule,
    use_mean_model: bool,
) -> dict:
    """
    Sample weights from the guide using Predictive.

    Parameters:
    ----------
    guide_pyro: The Pyro guide
    use_mean_model: Whether to use mean model

    Returns:
    -------
    Dictionary where each key is a weight name and value is a tensor of shape
    [num_samples, original_weight_shape]
    """
    if use_mean_model:
        return None

    batch_size = 10

    dummy_input = {
        "embeddings": torch.randn(batch_size, 320, 150),
        "initial_count": torch.ones(batch_size, 1),
        "initial_total": torch.ones(batch_size, 1),
        "selected_count": torch.ones(batch_size, 1),
        "selected_total": torch.ones(batch_size, 1),
        "total_sequences_in_experiment": batch_size,
    }

    with poutine.trace() as guide_trace:
        guide_pyro(dummy_input)

    # Get sampled values from trace
    sampled_values_from_trace = {
        name: site["value"]
        for name, site in guide_trace.trace.nodes.items()
        if site["type"] == "sample" and "predictor" in name
    }
    return sampled_values_from_trace


def from_pyro_to_pytorch_baseline_cnn_scorer(
    model_probabilistic: PyroModule,
    guide_probabilistic: AutoDiagonalNormal,
    cfg: DictConfig,
    weights: dict = None,
) -> DeterministicBayesianCNNCounterpart:
    """Transform a trained Bayesian model into a deterministic PyTorch model.

    This function extracts the posterior means from the trained guide and the
    running statistics from the trained probabilistic model's BatchNorm layers.

    Parameters
    ----------
    model_probabilistic : PyroModule
        The trained probabilistic model (e.g., PhageDisplayPoissonRegressor).
    guide_probabilistic : AutoDiagonalNormal
        The trained AutoGuide corresponding to the probabilistic model.
    cfg : DictConfig
        Configuration dictionary.
    weights : dict
        Optional weights to use instead of guide median

    Returns
    -------
    DeterministicBayesianCNNCounterpart
        The deterministic model with parameters set to the posterior means.
    """
    # Instantiate the deterministic model
    cnn_config = create_cnn_config(cfg)
    affinity_predictor = DeterministicBayesianCNNCounterpart(cnn_config)

    # Extract and process parameters
    if weights is None:
        print("Using the mean model")
        # median = mean of the posterior distribution for a normal distribution
        params = guide_probabilistic.median()
    else:
        params = weights
    cleaned_mean_params = clean_parameter_keys(params)
    batch_norm_stats = extract_batch_norm_stats(model_probabilistic)

    # Combine and load state dict
    final_state_dict = combine_state_dicts(cleaned_mean_params, batch_norm_stats)
    load_state_dict_with_logging(affinity_predictor, final_state_dict)

    return affinity_predictor


def clean_parameter_keys(mean_params: dict) -> dict:
    """Rename keys to match the deterministic model's state_dict."""
    return {
        key.replace("_sequence_affinity_predictor.", "")
        if key.startswith("_sequence_affinity_predictor.")
        else key: value
        for key, value in mean_params.items()
    }


def extract_batch_norm_stats(model_probabilistic: PyroModule) -> dict:
    """Extract BatchNorm statistics from the trained probabilistic model."""
    batch_norm_stats = {}
    for name, module in model_probabilistic.named_modules():
        if isinstance(module, torch.nn.BatchNorm1d):
            cleaned_name = name.replace("_sequence_affinity_predictor.", "")

            # Extract buffers
            batch_norm_stats[
                f"{cleaned_name}.running_mean"
            ] = module.running_mean.clone()
            batch_norm_stats[f"{cleaned_name}.running_var"] = module.running_var.clone()
            batch_norm_stats[
                f"{cleaned_name}.num_batches_tracked"
            ] = module.num_batches_tracked.clone()

    return batch_norm_stats


def combine_state_dicts(cleaned_mean_params: dict, batch_norm_stats: dict) -> dict:
    """Merge parameter and BatchNorm state dictionaries."""
    final_state_dict = cleaned_mean_params.copy()
    final_state_dict.update(batch_norm_stats)
    return final_state_dict


def load_state_dict_with_logging(
    model: DeterministicBayesianCNNCounterpart, state_dict: dict
):
    """Load state dict into model with logging for missing/unexpected keys."""
    missing_keys, unexpected_keys = model.load_state_dict(state_dict)

    if missing_keys:
        print(f"Missing keys (ignored): {missing_keys}")
    if unexpected_keys:
        print(f"Unexpected keys (ignored): {unexpected_keys}")
