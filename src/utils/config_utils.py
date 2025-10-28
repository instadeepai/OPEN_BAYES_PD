"""Configuration utilities for handling training parameters."""

import pyro
import torch
from omegaconf import DictConfig


def get_training_params(
    cfg: DictConfig, embedding_size: tuple[int, int], device: torch.device = None
) -> dict:
    """Extract training parameters from configuration with default values.

    Parameters:
        cfg: Configuration object containing model training parameters
        device: Device object to include in parameters (optional)

    Returns:
        Dictionary containing training parameters with defaults if keys are missing
    """
    layers_size = cfg.model.fitness_predictor_mlp.layers_size
    list_of_channels = cfg.model.fitness_predictor_cnn.list_of_channels

    layers_size[0] = embedding_size[0]
    list_of_channels[0] = embedding_size[0]

    architecture_dict = {
        "MLP": {
            "layers_size": layers_size,
        },
        "CNN": {
            "list_of_channels": list_of_channels,
        },
        "Transformer": {
            "num_embeddings": embedding_size[0],
            "sequence_length": embedding_size[1],
        },
        "HybridCNN": {
            "list_of_channels": list_of_channels,
        },
        "DoubleCNNWithNegative": {
            "list_of_channels": list_of_channels,
            "number_of_models": getattr(
                cfg.model.training, "number_of_models", "distinct"
            ),
        },
        "CNNWithBoltzmann": {
            "list_of_channels": list_of_channels,
            "number_of_models": getattr(
                cfg.model.training, "number_of_models", "distinct"
            ),
        },
    }

    training_params = {
        "dropout_value": getattr(cfg.model.training, "dropout_value", 0.1),
        "activation_name": getattr(cfg.model.training, "activation_name", "tanh"),
        "prior_scale": getattr(cfg.model.training, "prior_scale", 2.0),
        "device": device if device else torch.device("cpu"),
    }

    return {
        **training_params,
        **architecture_dict[cfg.model.fitness_predictor_architecture],
    }


def get_model_config_dict(
    cfg: DictConfig,
    architecture: str,
    embedding_size: tuple[int, int],
    device: torch.device = None,
) -> dict:
    """Create complete model configuration dictionary.

    Parameters:
        cfg: Configuration object
        architecture: Model architecture ('MLP' or 'CNN')
        device: Device object to include in parameters (optional)

    Returns:
        Complete configuration dictionary ready for model instantiation
    """
    # Get base model configuration
    if architecture == "MLP":
        config_dict = dict(cfg.model.fitness_predictor_mlp)
    elif architecture in [
        "CNN",
        "DoubleCNNWithNegative",
        "CNNWithBoltzmann",
        "HybridCNN",
    ]:
        config_dict = dict(cfg.model.fitness_predictor_cnn)
    elif architecture == "Transformer":
        config_dict = dict(cfg.model.fitness_predictor_transformer)
    else:
        raise ValueError(f"Unknown architecture: {architecture}")

    # Add training parameters
    training_params = get_training_params(cfg, embedding_size, device)
    config_dict.update(training_params)

    return config_dict


def count_bayesian_model_parameters(model) -> dict:
    """Count parameters in a Bayesian (Pyro) model.

    Parameters:
        model: Pyro/Bayesian model

    Returns:
        Dictionary with parameter counts and details
    """
    # Count standard PyTorch parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Count Pyro parameters (variational parameters)
    param_store = pyro.get_param_store()

    # Get all parameters that belong to this model
    model_pyro_params = 0
    pyro_param_names = []

    for name, param in param_store.items():
        if hasattr(param, "numel"):
            model_pyro_params += param.numel()
            pyro_param_names.append(name)

    non_trainable_params = total_params - trainable_params

    return {
        "pytorch_total_parameters": total_params,
        "pytorch_trainable_parameters": trainable_params,
        "pytorch_non_trainable_parameters": non_trainable_params,
        "pyro_parameters": model_pyro_params,
        "pyro_param_names": pyro_param_names,
        "total_all_parameters": total_params + model_pyro_params,
        "pytorch_params_millions": total_params / 1e6,
        "pyro_params_millions": model_pyro_params / 1e6,
        "total_params_millions": (total_params + model_pyro_params) / 1e6,
    }


def count_model_parameters(model: torch.nn.Module) -> dict:
    """Count the number of parameters in a PyTorch model.

    Parameters:
        model: PyTorch model

    Returns:
        Dictionary with parameter counts and details
    """
    # Check if it's a Pyro model
    if hasattr(model, "__module__") and "pyro" in str(type(model)).lower():
        return count_bayesian_model_parameters(model)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params

    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "non_trainable_parameters": non_trainable_params,
        "total_params_millions": total_params / 1e6,
        "trainable_params_millions": trainable_params / 1e6,
    }


def print_model_summary(model, model_name: str = "Bayesian Model") -> str:
    """Print a summary of Bayesian model parameters and return the summary string.

    Parameters:
        model: Pyro/Bayesian model
        model_name: Name to display for the model

    Returns:
        str: The formatted summary string
    """
    params_info = count_bayesian_model_parameters(model)

    # Build the summary string
    summary_lines = [
        f"\n{'=' * 60}",
        f"{model_name.upper()} PARAMETER SUMMARY",
        f"{'=' * 60}",
        "PyTorch Parameters:",
        f"  Total: {params_info['pytorch_total_parameters']:,}",
        f"  Trainable: {params_info['pytorch_trainable_parameters']:,}",
        f"  Non-trainable: {params_info['pytorch_non_trainable_parameters']:,}",
        "\nPyro Parameters:",
    ]

    if params_info["pyro_parameters"] > 0:
        summary_lines.extend(
            [
                f"  Variational params: {params_info['pyro_parameters']:,}",
                f"  Param names: {len(params_info['pyro_param_names'])}",
            ]
        )
    else:
        summary_lines.append(
            "  Variational params: 0 (not initialized - run model first)"
        )

    summary_lines.extend(
        [
            "\nTotals:",
            f"  All parameters: {params_info['total_all_parameters']:,}",
            f"  PyTorch (millions): {params_info['pytorch_params_millions']:.2f}M",
        ]
    )

    if params_info["pyro_parameters"] > 0:
        summary_lines.extend(
            [
                f"  Pyro (millions): {params_info['pyro_params_millions']:.2f}M",
                f"  Total (millions): {params_info['total_params_millions']:.2f}M",
            ]
        )
    else:
        summary_lines.extend(
            [
                "  Pyro (millions): 0.00M (not initialized)",
                f"  Total (millions): {params_info['pytorch_params_millions']:.2f}M",
            ]
        )

    summary_lines.append(f"{'=' * 60}")

    # Create the complete summary string
    summary_string = "\n".join(summary_lines)

    # Print the summary
    print(summary_string)

    # Return the summary string for logging/storage
    return summary_string
