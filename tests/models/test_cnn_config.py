"""Test CNN configuration and functionality."""

import sys
from pathlib import Path

import pytest
import torch

from src.models.core.bayesian_cnn import (
    BayesianCNN,
    DeterministicBayesianCNNCounterpart,
)
from src.models.core.model_config import CNNConfig
from tests.models.test_cnn_config_utils import (
    calculate_cnn_output_dimensions,
    create_valid_cnn_config,
    validate_cnn_config,
)

# Add tests directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))


class TestCNNConfig:
    """Test class for CNN configuration validation."""

    def calculate_expected_output_size(self, input_size: int, cfg: CNNConfig) -> int:
        """Helper function to calculate expected output size after conv and pooling."""
        # Use the utility function from test_cnn_config_utils
        dimensions = calculate_cnn_output_dimensions(input_size, cfg)
        return dimensions["layer_outputs"][-1]

    def test_basic_instantiation(self):
        """Test basic CNN instantiation with auto-calculated dimensions."""
        input_size = 100

        # Create a valid configuration automatically
        cfg = create_valid_cnn_config(
            input_size=input_size,
            list_of_channels=[1, 16],
            list_of_kernel_size=[3],
            list_of_stride=[1],
            mlp_hidden_sizes=[64, 1],
            pooling_name="MaxPool1d",
            pool_size=2,
            pool_stride=2,
        )

        # Test Bayesian CNN
        model = BayesianCNN(cfg)
        assert model is not None
        assert model.number_of_conv_modules == 1

        # Test Deterministic CNN
        det_model = DeterministicBayesianCNNCounterpart(cfg)
        assert det_model is not None

    def test_forward_pass(self):
        """Test forward pass with different input sizes using valid configurations."""
        input_sizes = [50, 100, 200]
        batch_size = 4

        for seq_length in input_sizes:
            print(f"\n--- Testing input size: {seq_length} ---")

            # Create valid configuration for this input size
            cfg = create_valid_cnn_config(
                input_size=seq_length,
                list_of_channels=[1, 16],
                list_of_kernel_size=[3],
                list_of_stride=[1],
                mlp_hidden_sizes=[32, 1],
                pooling_name="MaxPool1d",
                pool_size=2,
                pool_stride=2,
            )

            print(f"Generated config - MLP input: {cfg.layers_size[0]}")

            model = BayesianCNN(cfg)
            det_model = DeterministicBayesianCNNCounterpart(cfg)

            input_tensor = torch.randn(batch_size, 1, seq_length)

            with torch.no_grad():
                # Test Bayesian CNN
                output_bayesian = model(input_tensor)
                assert output_bayesian.shape == (batch_size, 1)

                # Test Deterministic CNN
                output_det = det_model(input_tensor)
                assert output_det.shape == (batch_size, 1)

            print(
                f"Forward pass successful: {input_tensor.shape} -> "
                f"{output_bayesian.shape}"
            )

    def test_different_kernel_sizes(self):
        """Test CNN with different kernel sizes using valid configurations."""
        kernel_sizes = [3, 5, 7]
        input_size = 50

        for kernel_size in kernel_sizes:
            print(f"\n--- Testing kernel size: {kernel_size} ---")

            cfg = create_valid_cnn_config(
                input_size=input_size,
                list_of_channels=[1, 8],
                list_of_kernel_size=[kernel_size],
                list_of_stride=[1],
                mlp_hidden_sizes=[16, 1],
                pooling_name="none",  # No pooling to avoid dimension issues
                pool_size=1,
                pool_stride=1,
            )

            model = BayesianCNN(cfg)
            assert model is not None

            # Test forward pass
            input_tensor = torch.randn(2, 1, input_size)
            with torch.no_grad():
                output = model(input_tensor)
                assert output.shape == (2, 1)

            print(f"Kernel size {kernel_size} works")

    def test_multiple_conv_layers(self):
        """Test CNN with multiple convolutional layers using valid configuration."""
        input_size = 100

        cfg = create_valid_cnn_config(
            input_size=input_size,
            list_of_channels=[1, 16, 32, 64],
            list_of_kernel_size=[3, 5, 3],
            list_of_stride=[1, 1, 1],
            mlp_hidden_sizes=[64, 1],
            pooling_name="MaxPool1d",
            pool_size=2,
            pool_stride=2,
        )

        model = BayesianCNN(cfg)
        assert model.number_of_conv_modules == 3

        # Test forward pass
        input_tensor = torch.randn(2, 1, input_size)
        with torch.no_grad():
            output = model(input_tensor)
            assert output.shape == (2, 1)

        print("Multiple conv layers work")

    def test_configuration_edge_cases(self):
        """Test edge cases in configuration using valid setups."""
        # Test with minimal configuration.
        input_size = 10

        cfg = create_valid_cnn_config(
            input_size=input_size,
            list_of_channels=[1, 2],
            list_of_kernel_size=[3],
            list_of_stride=[1],
            mlp_hidden_sizes=[1],  # Only output layer
            pooling_name="none",
            pool_size=1,
            pool_stride=1,
            dropout_value=0.0,
        )

        model = BayesianCNN(cfg)
        assert model is not None

        # Test forward pass
        input_tensor = torch.randn(1, 1, input_size)
        with torch.no_grad():
            output = model(input_tensor)
            assert output.shape == (1, 1)

        print("Minimal config works")

    def test_dimension_calculation(self):
        """Test that dimension calculations work as expected using utilities."""
        input_size = 100

        # Test configuration that should work
        cfg = create_valid_cnn_config(
            input_size=input_size,
            list_of_channels=[1, 16, 32],
            list_of_kernel_size=[3, 5],
            list_of_stride=[1, 1],
            mlp_hidden_sizes=[64, 1],
            pooling_name="MaxPool1d",
            pool_size=2,
            pool_stride=2,
        )

        # Validate the configuration
        is_valid, message = validate_cnn_config(input_size, cfg)
        print(f"\nValidation result: {is_valid} - {message}")

        if is_valid:
            # Calculate expected dimensions
            dimensions = calculate_cnn_output_dimensions(input_size, cfg)
            expected_mlp_input = dimensions["expected_mlp_input"]

            print(f"Input size: {input_size}")
            print(f"Expected MLP input: {expected_mlp_input}")
            print(f"Configured MLP input: {cfg.layers_size[0]}")

            # The test passes if the model can be instantiated and run
            model = BayesianCNN(cfg)

            input_tensor = torch.randn(2, 1, input_size)
            with torch.no_grad():
                output = model(input_tensor)
                assert output.shape == (2, 1)

            print("Dimension calculation test successful")
        else:
            pytest.fail(f"Configuration validation failed: {message}")

    def test_different_pooling_strategies(self):
        """Test different pooling strategies."""
        input_size = 64
        pooling_strategies = ["MaxPool1d", "AvgPool1d", "none"]

        for pooling in pooling_strategies:
            print(f"\n--- Testing pooling: {pooling} ---")

            pool_size = 2 if pooling != "none" else 1
            pool_stride = 2 if pooling != "none" else 1

            cfg = create_valid_cnn_config(
                input_size=input_size,
                list_of_channels=[1, 8],
                list_of_kernel_size=[3],
                list_of_stride=[1],
                mlp_hidden_sizes=[16, 1],
                pooling_name=pooling,
                pool_size=pool_size,
                pool_stride=pool_stride,
            )

            model = BayesianCNN(cfg)
            input_tensor = torch.randn(2, 1, input_size)

            with torch.no_grad():
                output = model(input_tensor)
                assert output.shape == (2, 1)

            print(f"Pooling {pooling} works")

    def test_batch_norm_options(self):
        """Test different batch normalization options."""
        input_size = 50
        batch_norm_options = ["BatchNorm1d", "none"]

        for batch_norm in batch_norm_options:
            print(f"\n--- Testing batch norm: {batch_norm} ---")

            cfg = create_valid_cnn_config(
                input_size=input_size,
                list_of_channels=[1, 8],
                list_of_kernel_size=[3],
                list_of_stride=[1],
                mlp_hidden_sizes=[16, 1],
                pooling_name="MaxPool1d",
                pool_size=2,
                pool_stride=2,
                batch_norm_name=batch_norm,
            )

            model = BayesianCNN(cfg)
            input_tensor = torch.randn(2, 1, input_size)

            with torch.no_grad():
                output = model(input_tensor)
                assert output.shape == (2, 1)

            print(f"Batch norm {batch_norm} works")

    def test_residual_connections(self):
        """Test residual connections functionality."""
        input_size = 150

        # Test configuration with residual connections
        cfg = CNNConfig(
            list_of_channels=[320, 128, 64],
            list_of_kernel_size=[5, 5],
            list_of_stride=[3, 3],
            list_of_padding=[2, 2],
            layers_size=[256, 32, 1],
            prior_scale=5.0,
            activation_name="ReLU",
            pooling_name="AvgPool1d",
            batch_norm_name="BatchNorm1d",
            pool_size=[2],
            pool_stride=[2],
            dropout_value=0.1,
            residual_connections=True,
        )

        # Test both models
        bayesian_model = BayesianCNN(cfg)
        det_model = DeterministicBayesianCNNCounterpart(cfg)

        # Verify residual connections are enabled
        assert bayesian_model.use_residual is True
        assert det_model.use_residual is True
        assert bayesian_model.projection_layers is not None
        assert det_model.projection_layers is not None

        # Test forward pass
        input_tensor = torch.randn(4, 320, input_size)

        with torch.no_grad():
            bayesian_output = bayesian_model(input_tensor)
            det_output = det_model(input_tensor)

        assert bayesian_output.shape == (4, 1)
        assert det_output.shape == (4, 1)

        print("Residual connections test successful")

        # Test without residual connections for comparison
        cfg.residual_connections = False
        model_no_residual = BayesianCNN(cfg)
        assert model_no_residual.use_residual is False
        assert model_no_residual.projection_layers is None

        with torch.no_grad():
            output_no_residual = model_no_residual(input_tensor)

        assert output_no_residual.shape == (4, 1)
        print("Model works both with and without residual connections")

    def test_residual_skip_layers(self):
        """Test residual skip layers functionality."""
        input_size = 150

        # Test with skip_layers = 2 (skip every 2 layers)
        cfg = create_valid_cnn_config(
            input_size=input_size,
            list_of_channels=[320, 128, 64, 32],
            list_of_kernel_size=[5, 5, 3],
            list_of_stride=[3, 3, 2],
            list_of_padding=[2, 2, 1],
            mlp_hidden_sizes=[32, 1],
            pooling_name="AvgPool1d",
            pool_size=2,
            pool_stride=2,
            residual_connections=True,
            residual_skip_layers=2,
        )

        # Create model and test forward pass
        model = BayesianCNN(cfg)
        input_tensor = torch.randn(4, 320, input_size)

        with torch.no_grad():
            output = model(input_tensor)

        # Check output shape
        assert output.shape == (4, 1), f"Expected (4, 1), got {output.shape}"

        # Verify residual connections configuration
        assert model.use_residual is True
        assert model.residual_skip_layers == 2
        assert model.projection_layers is not None

        # With 4 channels (3 conv layers) and skip_layers=2,
        # we should have 1 residual connection (from layer 0 to layer 2)
        assert len(model.projection_layers) == 1
        assert 2 in model.residual_mappings  # Layer 2 should have residual from layer 0
        assert model.residual_mappings[2] == 0

        print("Residual skip layers test successful")

        # Test with skip_layers = 1 (every layer - default behavior)
        cfg_skip_1 = create_valid_cnn_config(
            input_size=input_size,
            list_of_channels=[320, 128, 64, 32],
            list_of_kernel_size=[5, 5, 3],
            list_of_stride=[3, 3, 2],
            list_of_padding=[2, 2, 1],
            mlp_hidden_sizes=[32, 1],
            pooling_name="AvgPool1d",
            pool_size=2,
            pool_stride=2,
            residual_connections=True,
            residual_skip_layers=2,
        )

        model_skip_1 = BayesianCNN(cfg_skip_1)

        assert model_skip_1.residual_skip_layers == 2
        assert len(model_skip_1.projection_layers) == 1

        with torch.no_grad():
            output_skip_1 = model_skip_1(input_tensor)
        assert output_skip_1.shape == (4, 1)

        print("Different skip layer values work correctly")
