"""Test current CNN configuration from baseline.yaml."""

import sys
from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig

from src.models.core.bayesian_cnn import (
    BayesianCNN,
    DeterministicBayesianCNNCounterpart,
)
from src.models.core.model_config import CNNConfig
from src.utils.config_utils import get_model_config_dict
from tests.models.test_cnn_config_utils import (
    calculate_cnn_output_dimensions,
    print_config_summary,
    validate_cnn_config,
)

# Add tests directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))


class TestCurrentCNNConfig:
    """Test class to validate current CNN configuration."""

    @classmethod
    def setup_class(cls):
        """Setup class to initialize Hydra with config path."""
        # Clear any existing Hydra instance
        GlobalHydra.instance().clear()

        # Get the absolute path to the config directory
        cls.config_dir = Path(__file__).parent.parent.parent / "config"
        cls.config_dir = cls.config_dir.resolve()

    def teardown_method(self):
        """Clean up Hydra after each test."""
        if GlobalHydra.instance().is_initialized():
            GlobalHydra.instance().clear()

    def load_config(self, config_name: str = "default") -> DictConfig:
        """
        Load configuration using Hydra.

        Parameters:
        ----------
        config_name (str):
            name of the config file to load.

        Returns:
        -------
        DictConfig: loaded configuration.
        """

        with initialize_config_dir(config_dir=str(self.config_dir), version_base=None):
            cfg = compose(config_name=config_name)
        return cfg

    def test_load_baseline_config(self):
        """Test loading of the baseline configuration."""
        cfg = self.load_config("default")

        assert cfg is not None
        assert hasattr(cfg, "model")
        assert hasattr(cfg.model, "fitness_predictor_cnn")

        cnn_config = cfg.model.fitness_predictor_cnn
        print("\n=== Loaded CNN Configuration ===")
        print(f"Channels: {cnn_config.list_of_channels}")
        print(f"Kernel sizes: {cnn_config.list_of_kernel_size}")
        print(f"Strides: {cnn_config.list_of_stride}")
        print(f"MLP layer sizes: {cnn_config.layers_size}")
        print(f"Pooling: {cnn_config.pooling_name}")

    def _validate_config_helper(self, cfg) -> tuple[bool, str, CNNConfig, int]:
        """Helper function to validate config and return common values."""
        cnn_config_dict = get_model_config_dict(cfg, "CNN", cfg.model.embedding_size)
        # delete possible num_embeddings parameter
        cnn_config_dict.pop("num_embeddings", None)
        cnn_config_dict.pop("sequence_length", None)
        cnn_config = CNNConfig(**cnn_config_dict)
        input_size = cfg.model.cnn_sequence_length

        is_valid, error_message = validate_cnn_config(input_size, cnn_config)

        return is_valid, error_message, cnn_config, input_size

    def test_validate_current_cnn_config(self):
        """Test if current CNN configuration is mathematically valid."""
        cfg = self.load_config("default")

        print("\n=== CNN Configuration Validation ===")
        is_valid, error_message, cnn_config, input_size = self._validate_config_helper(
            cfg
        )

        print_config_summary(input_size, cnn_config)

        # Show detailed calculation steps
        dimensions = calculate_cnn_output_dimensions(input_size, cnn_config)

        if not is_valid:
            expected_mlp = dimensions["expected_mlp_input"]
            configured_mlp = dimensions["configured_mlp_input"]
            fail_message = f"""
CURRENT CNN CONFIGURATION IS INVALID!

Error: {error_message}

Problem: {dimensions}

Configuration file: config/model/baseline.yaml
Expected: layers_size[0] should be {expected_mlp}
Actual: layers_size[0] is {configured_mlp}

To fix: Update the layers_size[0] parameter in baseline.yaml
            """
            pytest.fail(fail_message.strip())

        print("Current configuration is VALID!")

    def _test_model_instantiation_helper(self, cnn_config, input_size, cfg):
        """Helper to test model instantiation."""
        embedding_size = cfg.model.embedding_size
        batch_size = 4

        bayesian_model = BayesianCNN(cnn_config)
        det_model = DeterministicBayesianCNNCounterpart(cnn_config)

        input_tensor = torch.randn(batch_size, embedding_size[0], input_size)

        with torch.no_grad():
            try:
                bayesian_output = bayesian_model(input_tensor)
                det_output = det_model(input_tensor)
                return bayesian_output, det_output
            except RuntimeError as e:
                raise RuntimeError(f"Forward pass failed: {e}") from e

    def test_instantiate_current_cnn_models(self):
        """Test instantiation of CNN models with current configuration."""
        cfg = self.load_config("default")

        print("\n=== Model Instantiation Test ===")
        is_valid, error_message, cnn_config, input_size = self._validate_config_helper(
            cfg
        )

        if not is_valid:
            fail_message = f"""
CANNOT INSTANTIATE MODEL - CONFIGURATION IS INVALID!

Configuration problem: {error_message}

Expected behavior: Models should not be instantiated with invalid
configurations. This is the ROOT CAUSE of training failures with CNN
architecture.

To fix: Update config/model/baseline.yaml with corrected parameters.
            """
            pytest.fail(fail_message.strip())

        try:
            bayesian_output, det_output = self._test_model_instantiation_helper(
                cnn_config, input_size, cfg
            )
            print(f"Bayesian forward pass successful: {bayesian_output.shape}")
            print(f"Deterministic forward pass successful: {det_output.shape}")

        except RuntimeError as e:
            print(f"Error during instantiation: {e}")
            print_config_summary(input_size, cnn_config)
            pytest.fail(f"MODEL INSTANTIATION FAILED! Error: {e}")

    def test_suggest_config_fix(self):
        """Test configuration validation and show if fixes are needed."""
        cfg = self.load_config("default")

        print("\n=== Configuration Validation ===")
        is_valid, error_message, cnn_config, input_size = self._validate_config_helper(
            cfg
        )

        print_config_summary(input_size, cnn_config)

        if not is_valid:
            print("\nCURRENT CONFIGURATION IS INVALID!")
            print(f"Error: {error_message}")

            dimensions = calculate_cnn_output_dimensions(input_size, cnn_config)
            expected_mlp_input = dimensions["expected_mlp_input"]

            print("\nSuggested fix:")
            print(
                f"Update layers_size[0] from {cnn_config.layers_size[0]} "
                f"to {expected_mlp_input}"
            )
        else:
            print("Current configuration is valid!")

    def _test_single_input_size(self, input_size, cnn_config, embedding_size):
        """Helper to test a single input size."""
        try:
            analysis = validate_cnn_config(input_size, cnn_config)
            if not analysis[0]:
                return False, f"Configuration not valid: {analysis[1]}"

            model = BayesianCNN(cnn_config)
            input_tensor = torch.randn(2, embedding_size[0], input_size)

            with torch.no_grad():
                output = model(input_tensor)
                return True, f"{input_tensor.shape} -> {output.shape}"

        except (RuntimeError, ValueError, IndexError) as e:
            return False, f"Forward pass failed: {e}"

    def test_current_config_with_different_input_sizes(self):
        """Test current configuration with different input sizes."""
        cfg = self.load_config("default")
        cnn_config_dict = get_model_config_dict(cfg, "CNN", cfg.model.embedding_size)
        # delete possible num_embeddings parameter
        cnn_config_dict.pop("num_embeddings", None)
        cnn_config_dict.pop("sequence_length", None)
        cnn_config = CNNConfig(**cnn_config_dict)
        test_sizes = [cfg.model.cnn_sequence_length]
        embedding_size = cfg.model.embedding_size

        print("\n=== Test with Different Input Sizes ===")
        print(f"Embedding size: {embedding_size}")

        failed_sizes = []
        success_sizes = []

        for input_size in test_sizes:
            print(f"\n{'='*60}")
            print(f"TESTING INPUT SIZE: {input_size}")
            print(f"{'='*60}")

            success, message = self._test_single_input_size(
                input_size, cnn_config, embedding_size
            )

            if success:
                print(f"Forward pass successful: {message}")
                success_sizes.append(input_size)
            else:
                print(message)
                failed_sizes.append(input_size)

        if failed_sizes:
            fail_message = f"""
CURRENT CONFIGURATION FAILS FOR MULTIPLE INPUT SIZES!

Failed sizes: {failed_sizes}
Success sizes: {success_sizes}

This indicates the current CNN configuration in baseline.yaml is
fundamentally broken and needs to be redesigned to work with the expected
input dimensions.
            """
            pytest.fail(fail_message.strip())
        else:
            print("Configuration works for all tested input sizes")

    @pytest.mark.parametrize("config_type", ["baseline"])
    def test_all_model_configs(self, config_type):
        """Test all available model configurations."""
        # Load config by overriding the model type
        with initialize_config_dir(config_dir=str(self.config_dir), version_base=None):
            cfg = compose(config_name="default", overrides=[f"model={config_type}"])

        print(f"\n=== Test model config: {config_type} ===")

        if hasattr(cfg.model, "fitness_predictor_cnn"):
            cnn_config_dict = get_model_config_dict(
                cfg, "CNN", cfg.model.embedding_size
            )
            # delete possible num_embeddings parameter
            cnn_config_dict.pop("num_embeddings", None)
            cnn_config_dict.pop("sequence_length", None)
            cnn_config = CNNConfig(**cnn_config_dict)
            input_size = cfg.model.cnn_sequence_length

            # Perform detailed analysis
            analysis = validate_cnn_config(input_size, cnn_config)
            print_config_summary(input_size, cnn_config)

            # FAIL if the configuration is invalid
            if not analysis[0]:
                error_message = analysis[1]
                fail_message = f"""
MODEL CONFIG '{config_type}' HAS INVALID CNN CONFIGURATION!

Issues found:
  • {error_message}

This model configuration cannot be used for training until fixed.
                """
                pytest.fail(fail_message.strip())
            else:
                print(f"Model config '{config_type}' is valid")
        else:
            pytest.skip(f"Configuration {config_type} doesn't have CNN")
