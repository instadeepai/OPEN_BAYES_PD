# pylint: disable=W0621,R0914,R0801,W0611,duplicate-code

"""Test module for null model functionality."""

import os
import sys
from pathlib import Path

import numpy as np
import pytest
from hydra import compose, initialize

from src.data_processing.prepare_dataset import prepare_dataset_for_training
from src.utils.calculate_null_model import calculate_null_model_correlation
from src.utils.device import select_device
from tests.conftest import _random_seeder  # noqa: F401
from tests.conftest import setup_and_destroy_experiment_dir  # noqa: F811

# Add tests directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


@pytest.mark.usefixtures("_random_seeder")
def test_null_model(setup_and_destroy_experiment_dir):
    """Test null model correlation calculation for both distributions."""
    print(f"\n{'=' * 70}")
    print("TESTING NULL MODEL WITH BOTH DISTRIBUTIONS")
    print(f"{'=' * 70}")

    distributions = [
        "src.models.baseline_model.PhageDisplayPoissonRegressor",
        "src.models.baseline_model.PhageDisplayMultinomialRegressor",
    ]

    for distribution in distributions:
        print(f"\n--- Testing {distribution.upper()} distribution ---")

        # Prepare configuration with null model enabled and specific distribution
        with initialize(version_base=None, config_path="../config"):
            cfg = compose(
                config_name="default.yaml",
                overrides=[
                    "model=baseline",
                    "model.fitness_predictor_architecture=MLP",
                    "model.training.is_null_model=true",
                    "model.training.epochs=1",
                    f"model.training.distribution={distribution}",
                ],
            )

        device = select_device(cfg)
        csv_path = Path(__file__).parent.parent / "data" / "csv" / "test.csv"
        batches_dir = setup_and_destroy_experiment_dir

        # Prepare dataloaders
        (
            _,
            _,
            train_dataloader,
            valid_dataloader,
            _,
            _,
            _,
        ) = prepare_dataset_for_training(
            csv_path=csv_path,
            cfg=cfg,
            batches_path=batches_dir,
            device=device,
            test_data_path=None,
        )

        # Calculate correlations
        train_corr = {}
        valid_corr = {}
        for experiment in cfg.model.splits.training_pair:
            print(f"Calculating null model correlation for {experiment}")
            train_corr[experiment] = calculate_null_model_correlation(
                train_dataloader, experiment, cfg, seed=42
            )
        for experiment in cfg.model.splits.validation_pair:
            print(f"Calculating null model correlation for {experiment}")
            valid_corr[experiment] = calculate_null_model_correlation(
                valid_dataloader, experiment, cfg, seed=42
            )

        print(f"Training correlation ({distribution}): {train_corr}")
        print(f"Validation correlation ({distribution}): {valid_corr}")

        # Test training correlations
        for experiment, corr in train_corr.items():
            assert (
                -1.0 <= corr <= 1.0
            ), f"Training correlation {corr} not in [-1, 1] for {distribution}"
            assert not np.isnan(
                corr
            ), f"Training correlation is NaN for {experiment} in {distribution}"
            assert np.isfinite(
                corr
            ), f"Training correlation is not finite for {experiment} in {distribution}"

        # Test validation correlations
        for experiment, corr in valid_corr.items():
            assert (
                -1.0 <= corr <= 1.0
            ), f"Validation correlation {corr} not in [-1, 1] for {distribution}"
            assert not np.isnan(
                corr
            ), f"Validation correlation is NaN for {experiment} in {distribution}"
            assert np.isfinite(
                corr
            ), f"Validation correlation is not finit for {experiment} in {distribution}"

        print(f"✓ {distribution.capitalize()} distribution test passed!")

    print("\n✓ All null model tests passed!")
