"""
Utility functions for CNN configuration validation and dimension calculation.
"""

from src.models.core.model_config import CNNConfig


def calculate_conv_output_size(
    input_size: int, kernel_size: int, stride: int = 1, padding: int = 1
) -> int:
    """Calculate output size after convolution operation."""
    return (input_size + 2 * padding - kernel_size) // stride + 1


def calculate_pool_output_size(
    input_size: int, pool_size: int, pool_stride: int
) -> int:
    """Calculate output size after pooling operation.

    If pool_size or pool_stride is 0, pooling is disabled and input_size is returned.
    """
    # If pool_size or pool_stride is 0, disable pooling
    if pool_size == 0 or pool_stride == 0:
        return input_size
    return (input_size - pool_size) // pool_stride + 1


def _get_layer_parameter(param_list: list, layer_idx: int, param_name: str) -> int:
    """Get parameter for specific layer, handling single value lists."""
    try:
        return param_list[layer_idx]
    except IndexError as e:
        if len(param_list) == 1:
            return param_list[0]
        msg = (
            f"Invalid {param_name} config: layer {layer_idx} "
            f"not found in {param_list}"
        )
        raise ValueError(msg) from e


def _process_conv_layer(current_size: int, cfg: CNNConfig, layer_idx: int) -> int:
    """Process a single convolutional layer and return output size."""
    kernel_size = _get_layer_parameter(
        cfg.list_of_kernel_size, layer_idx, "kernel_size"
    )
    stride = _get_layer_parameter(cfg.list_of_stride, layer_idx, "stride")
    padding = _get_layer_parameter(cfg.list_of_padding, layer_idx, "padding")

    return calculate_conv_output_size(
        current_size, kernel_size, stride, padding=padding
    )


def _process_pooling_layer(current_size: int, cfg: CNNConfig, layer_idx: int) -> int:
    """Process a single pooling layer and return output size.

    If pooling_name is "None" or "none" or pool parameters are 0, pooling is disabled.
    """
    if cfg.pooling_name.lower() in ["none", None]:
        return current_size

    pool_kernel = _get_layer_parameter(cfg.pool_size, layer_idx, "pool_size")
    pool_stride = _get_layer_parameter(cfg.pool_stride, layer_idx, "pool_stride")

    # If either pool parameter is 0, disable pooling for this layer
    if pool_kernel == 0 or pool_stride == 0:
        return current_size

    return calculate_pool_output_size(current_size, pool_kernel, pool_stride)


def calculate_cnn_output_dimensions(input_size: int, cfg: CNNConfig) -> dict:
    """Calculate CNN output dimensions step by step."""
    current_size = input_size
    layer_outputs = []

    # Process each convolutional layer
    for i in range(len(cfg.list_of_channels) - 1):
        # Convolution
        current_size = _process_conv_layer(current_size, cfg, i)
        layer_outputs.append(current_size)

        # Pooling
        current_size = _process_pooling_layer(current_size, cfg, i)
        layer_outputs.append(current_size)

        if current_size <= 0:
            break

    final_channels = cfg.list_of_channels[-1]
    expected_mlp_input = current_size * final_channels
    configured_mlp_input = cfg.layers_size[0] if cfg.layers_size else 0

    return {
        "final_conv_size": current_size,
        "expected_mlp_input": expected_mlp_input,
        "configured_mlp_input": configured_mlp_input,
        "is_valid": (current_size > 0 and expected_mlp_input == configured_mlp_input),
        "layer_outputs": layer_outputs,
    }


def suggest_mlp_input_size(input_size: int, cfg: CNNConfig) -> int:
    """Suggest correct MLP input size for CNN configuration."""
    dimensions = calculate_cnn_output_dimensions(input_size, cfg)
    return dimensions["expected_mlp_input"]


def create_valid_cnn_config(
    input_size: int,
    list_of_channels: list[int],
    list_of_kernel_size: list[int],
    *,  # Force keyword-only arguments after this
    list_of_stride: list[int] = None,
    mlp_hidden_sizes: list[int] = None,
    **kwargs,
) -> CNNConfig:
    """Create valid CNN configuration with calculated MLP input size."""

    # Default configuration parameters
    defaults = {
        "pooling_name": "MaxPool1d",
        "pool_size": 2,
        "pool_stride": 2,
        "output_size": 1,
        "prior_scale": 0.1,
        "activation_name": "ReLU",
        "batch_norm_name": "BatchNorm1d",
        "dropout_value": 0.1,
        "list_of_padding": [1],
        "residual_connections": False,
        "residual_skip_layers": 1,
        "device": None,
    }

    # Update defaults with provided kwargs
    config_params = {**defaults, **kwargs}

    # Create temporary config to calculate dimensions
    temp_cfg = CNNConfig(
        list_of_channels=list_of_channels,
        list_of_kernel_size=list_of_kernel_size,
        list_of_stride=list_of_stride,
        layers_size=[1],  # Temporary
        prior_scale=config_params["prior_scale"],
        activation_name=config_params["activation_name"],
        pooling_name=config_params["pooling_name"],
        batch_norm_name=config_params["batch_norm_name"],
        pool_size=[config_params["pool_size"]],
        pool_stride=[config_params["pool_stride"]],
        dropout_value=config_params["dropout_value"],
        residual_connections=config_params["residual_connections"],
        residual_skip_layers=config_params["residual_skip_layers"],
        device=config_params["device"],
        list_of_padding=config_params["list_of_padding"],
    )

    # Calculate correct MLP input size
    mlp_input_size = suggest_mlp_input_size(input_size, temp_cfg)
    layers_size = [mlp_input_size] + mlp_hidden_sizes + [config_params["output_size"]]

    # Create final config
    return CNNConfig(
        list_of_channels=list_of_channels,
        list_of_kernel_size=list_of_kernel_size,
        list_of_stride=list_of_stride,
        list_of_padding=config_params["list_of_padding"],
        layers_size=layers_size,
        prior_scale=config_params["prior_scale"],
        activation_name=config_params["activation_name"],
        pooling_name=config_params["pooling_name"],
        batch_norm_name=config_params["batch_norm_name"],
        pool_size=[config_params["pool_size"]],
        pool_stride=[config_params["pool_stride"]],
        dropout_value=config_params["dropout_value"],
        residual_connections=config_params["residual_connections"],
        residual_skip_layers=config_params["residual_skip_layers"],
        device=config_params["device"],
    )


def validate_cnn_config(input_size: int, cfg: CNNConfig) -> tuple[bool, str]:
    """Validate CNN configuration."""
    try:
        dimensions = calculate_cnn_output_dimensions(input_size, cfg)

        if dimensions["final_conv_size"] <= 0:
            return (False, f"Invalid size: {dimensions['final_conv_size']}")

        if not dimensions["is_valid"]:
            expected = dimensions["expected_mlp_input"]
            configured = dimensions["configured_mlp_input"]
            return (False, f"MLP input mismatch: expected {expected}, got {configured}")

        return True, "Valid configuration"

    except (IndexError, KeyError, ValueError) as e:
        return False, f"Validation error: {str(e)}"


def print_config_summary(input_size: int, cfg: CNNConfig):
    """Print concise CNN configuration summary."""
    print("\n" + "=" * 40)
    print("CNN CONFIGURATION SUMMARY")
    print("=" * 40)

    dimensions = calculate_cnn_output_dimensions(input_size, cfg)

    print(f"Input size: {input_size}")
    print(f"Channels: {cfg.list_of_channels}")
    print(f"Kernel sizes: {cfg.list_of_kernel_size}")
    print(f"Strides: {cfg.list_of_stride}")
    print(f"Pooling: {cfg.pooling_name}")

    print("\nOutput dimensions:")
    print(f"Final conv size: {dimensions['final_conv_size']}")
    print(f"Expected MLP input: {dimensions['expected_mlp_input']}")
    print(f"Configured MLP input: {dimensions['configured_mlp_input']}")
    print(f"MLP layers: {cfg.layers_size}")

    status = "VALID" if dimensions["is_valid"] else "INVALID"
    print(f"\nStatus: {status}")
    print("=" * 40)
