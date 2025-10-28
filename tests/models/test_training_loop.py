# pylint: disable=W0621,R0914,R0801,W0611,duplicate-code, R0913, R0917

"""Test module for training loop functionality."""

import os
import re
import sys
import tempfile
from io import StringIO
from pathlib import Path

import neptune
import pytest
from hydra import compose, initialize

from src.data_processing.prepare_dataset import prepare_dataset_for_training
from src.models.training_helpers import train_bnn
from src.utils.device import select_device
from tests.conftest import _random_seeder  # noqa: F401
from tests.conftest import setup_and_destroy_experiment_dir  # noqa: F401

# Add tests directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def _validate_training_progress(loss_values: list[float], loss_type: str):
    """Helper function to validate training progress."""
    if len(loss_values) >= 3:
        # Check if loss improves between first third and last third of training
        first_third = loss_values[: len(loss_values) // 3]
        last_third = loss_values[-len(loss_values) // 3 :]
        avg_early_loss = sum(first_third) / len(first_third)
        avg_late_loss = sum(last_third) / len(last_third)

        print(f"Early training average {loss_type} loss: {avg_early_loss:.3f}")
        print(f"Late training average {loss_type} loss: {avg_late_loss:.3f}")

        # Loss should improve (become less negative) or at least not get much worse
        loss_improvement = avg_late_loss - avg_early_loss
        print(f"{loss_type} loss improvement: {loss_improvement:.3f}")

        # Check that loss didn't get significantly worse (allow 10% degradation)
        max_degradation = abs(avg_early_loss) * 0.25
        assert loss_improvement > -max_degradation, (
            f"Training {loss_type} degraded too much: {loss_improvement:.3f} vs "
            f"max allowed: {-max_degradation:.3f}"
        )
        print("Training showed acceptable progress")


def _validate_loss_values(loss_values: list[float], loss_type: str):
    """Helper function to validate loss values are reasonable."""
    best_loss = max(loss_values)
    worst_loss = min(loss_values)

    # Check that losses are finite numbers
    assert all(
        abs(loss) < float("inf") for loss in loss_values
    ), "All loss values should be finite"
    assert not any(
        str(loss) == "nan" for loss in loss_values
    ), "No loss values should be NaN"
    print(f"All {loss_type} loss values are valid finite numbers")

    # Check that losses are in reasonable range (should be negative for ELBO)
    assert all(loss < 0 for loss in loss_values), f"{loss_type} loss should be negative"
    assert best_loss > -1e8, f"Best loss {best_loss} seems too extreme"
    assert worst_loss < 0, f"Worst loss {worst_loss} should be negative"
    print(f"Loss values in reasonable range: {worst_loss:.3f} to {best_loss:.3f}")


def _extract_loss_values(train_output, pattern_loss):
    """Helper function to extract loss values from training output."""
    loss_values = []
    for line in train_output.split("\n"):
        match_loss = re.search(pattern_loss, line)
        if match_loss:
            loss = float(match_loss.group(1))
            loss_values.append(loss)
    return loss_values


def _prepare_training_setup(temp_dir, model_type, architecture_type):
    """Helper function to prepare training setup and reduce local variables."""
    # Prepare configuration
    with initialize(version_base=None, config_path="../config"):
        cfg = compose(
            config_name="default.yaml",
            overrides=[
                f"model={model_type}",
                f"model.fitness_predictor_architecture={architecture_type}",
            ],
        )

    device = select_device(cfg)
    csv_path = Path(__file__).parent.parent / "data" / "csv" / "test.csv"
    batches_dir = temp_dir / "batches"
    batches_dir.mkdir()

    _, _, train_dataloader, valid_dataloader, _, _, _ = prepare_dataset_for_training(
        csv_path=csv_path,
        cfg=cfg,
        batches_path=batches_dir,
        device=device,
        test_data_path=None,
    )

    return cfg, train_dataloader, valid_dataloader


def _run_training_and_capture_output(cfg, temp_dir, train_dataloader, valid_dataloader):
    """Helper function to run training and capture output."""
    old_stdout = sys.stdout
    sys.stdout = string_stdout = StringIO()
    neptune_dummy = neptune.init_run(mode="debug")
    device = select_device(cfg)

    train_bnn(
        train_dataloader,
        valid_dataloader,
        None,
        cfg,
        temp_dir,
        None,
        neptune_dummy,
        device,
        cfg.model.embedding_size,
        kd_values=None,
    )
    neptune_dummy.stop()

    sys.stdout = old_stdout
    return string_stdout.getvalue()


def _run_complete_training_test(temp_dir, model_type, architecture_type):
    """Helper function to run complete training test and return results."""
    # Prepare training setup
    cfg, train_dataloader, valid_dataloader = _prepare_training_setup(
        temp_dir, model_type, architecture_type
    )

    print(f"\n=== TESTING TRAINING: {model_type} {architecture_type} ===")

    # Run training and capture output
    train_output = _run_training_and_capture_output(
        cfg, temp_dir, train_dataloader, valid_dataloader
    )

    return cfg, train_output


@pytest.mark.parametrize(
    (
        "architecture_type",
        "distribution",
        "batch_size",
        "estimate_method",
        "number_of_models",
    ),
    [
        (
            "Transformer",
            "src.models.baseline_model.PhageDisplayPoissonRegressor",
            "null",
            "global",
            "null",
        ),
        (
            "CNNWithBoltzmann",
            "src.models.baseline_model.PhageDisplayPoissonRegressor",
            "null",
            "global",
            "two",
        ),
        (
            "CNNWithBoltzmann",
            "src.models.baseline_model.PhageDisplayPoissonRegressor",
            "null",
            "global",
            "distinct",
        ),
        (
            "CNNWithBoltzmann",
            "src.models.baseline_model.PhageDisplayPoissonRegressor",
            "null",
            "global",
            "one",
        ),
        (
            "DoubleCNNWithNegative",
            "src.models.baseline_model.PhageDisplayPoissonRegressor",
            "null",
            "global",
            "one",
        ),
        (
            "DoubleCNNWithNegative",
            "src.models.baseline_model.PhageDisplayPoissonRegressor",
            "null",
            "global",
            "sequential_latent",
        ),
        (
            "DoubleCNNWithNegative",
            "src.models.baseline_model.PhageDisplayPoissonRegressor",
            "null",
            "global",
            "sequential_output",
        ),
        (
            "DoubleCNNWithNegative",
            "src.models.baseline_model.PhageDisplayPoissonRegressor",
            "null",
            "global",
            "distinct",
        ),
        (
            "MLP",
            "src.models.baseline_model.PhageDisplayPoissonRegressor",
            "null",
            "global",
            "null",
        ),
        (
            "CNN",
            "src.models.baseline_model.PhageDisplayPoissonRegressor",
            10,
            "N-1",
            "null",
        ),
        (
            "CNN",
            "src.models.baseline_model.PhageDisplayPoissonRegressor",
            10,
            "batch",
            "null",
        ),
        (
            "CNN",
            "src.models.baseline_model.PhageDisplayPoissonRegressor",
            "null",
            "moving_average",
            "null",
        ),
        (
            "CNN",
            "src.models.baseline_model.PhageDisplayPoissonRegressor",
            "null",
            "global",
            "null",
        ),
        (
            "CNN",
            "src.models.baseline_model.PhageDisplayPoissonRegressor",
            "null",
            "EMA",
            "null",
        ),
        (
            "CNN",
            "src.models.baseline_model.PhageDisplayMultinomialRegressor",
            "null",
            "global",
            "null",
        ),
    ],
)
@pytest.mark.usefixtures("_random_seeder")
def test_training_function_is_working_properly(
    setup_and_destroy_experiment_dir,  # noqa: F811
    architecture_type,
    distribution,
    batch_size,
    estimate_method,
    number_of_models,
):
    """Test that the training function works correctly with robust validation."""
    print(f"\n{'=' * 70}")
    print(
        f"TESTING TRAINING LOOP: {architecture_type} - "
        f"{distribution} - {estimate_method}"
    )
    print(f"{'=' * 70}")

    print(f"Temporary directory: {setup_and_destroy_experiment_dir}")

    # Prepare configuration
    with initialize(version_base=None, config_path="../config"):
        cfg = compose(
            config_name="default.yaml",
            overrides=[
                f"model.fitness_predictor_architecture={architecture_type}",
                f"model.training.distribution={distribution}",
                f"model.training.batch_size={batch_size}",
                f"model.training.estimate_method={estimate_method}",
                f"model.training.number_of_models={number_of_models}",
            ],
        )

    device = select_device(cfg)
    csv_path = Path(__file__).parent.parent / "data" / "csv" / "test.csv"
    batches_dir = setup_and_destroy_experiment_dir / "batches"
    batches_dir.mkdir()

    # Prepare dataset and run training
    _, _, train_dataloader, valid_dataloader, _, _, _ = prepare_dataset_for_training(
        csv_path=csv_path,
        cfg=cfg,
        batches_path=batches_dir,
        device=device,
        test_data_path=None,
    )

    train_output = _run_training_and_capture_output(
        cfg, setup_and_destroy_experiment_dir, train_dataloader, valid_dataloader
    )

    # Test 1: Model file is saved
    assert (
        setup_and_destroy_experiment_dir / "model.pth"
    ).exists(), "Model file should be saved"
    print("Model file saved successfully")

    # Test 2: All epochs run
    num_epochs = cfg.model.training.epochs
    for epoch_number in range(num_epochs):
        assert (
            f"------- EPOCH {epoch_number + 1} -------" in train_output
        ), f"Epoch {epoch_number + 1} should run"
    print(f"All {num_epochs} epochs completed")

    pattern_elbo_loss = r"Average training elbo loss: (-?\d+\.\d+)"
    pattern_reconstruction_loss = r"Average training reconstruction loss: (-?\d+\.\d+)"
    pattern_kl_loss = r"Average training kl loss: (-?\d+\.\d+)"

    # Test 3: Extract all loss values to check training progression
    elbo_loss_values = _extract_loss_values(train_output, pattern_elbo_loss)
    reconstruction_loss_values = _extract_loss_values(
        train_output, pattern_reconstruction_loss
    )
    kl_loss_values = _extract_loss_values(train_output, pattern_kl_loss)

    # Test 3: Verify we have sufficient loss values for all loss types
    assert len(elbo_loss_values) >= num_epochs, (
        f"Should have at least {num_epochs} ELBO loss values, "
        f"got {len(elbo_loss_values)}"
    )
    assert len(reconstruction_loss_values) >= num_epochs, (
        f"Should have at least {num_epochs} reconstruction loss values, "
        f"got {len(reconstruction_loss_values)}"
    )
    assert (
        len(kl_loss_values) >= num_epochs
    ), f"Should have at least {num_epochs} KL loss values, got {len(kl_loss_values)}"
    print(f"Found {len(elbo_loss_values)} loss values from training")

    # Test 4: Check that training is making progress
    _validate_training_progress(elbo_loss_values, "ELBO")

    # Test 5: Check loss values are reasonable (not NaN, not extreme)
    _validate_loss_values(elbo_loss_values, "ELBO")

    # Test 6: Check that "Best epoch" information is present
    assert "Best epoch:" in train_output, "Training should track best epoch"

    # Test 7: check that reconstruction and kl loss are not zero
    assert all(
        loss != 0 for loss in reconstruction_loss_values
    ), "Reconstruction loss should not be zero"
    assert all(loss != 0 for loss in kl_loss_values), "KL loss should not be zero"

    print("Best epoch tracking working")

    print(
        f"Training test passed for {architecture_type}-"
        f"{distribution}-{estimate_method}!"
    )


@pytest.mark.usefixtures("_random_seeder")
def test_training_with_different_epochs():
    """Test training with different number of epochs to verify scalability."""
    model_type = "baseline"
    architecture_type = "MLP"

    for num_epochs in [2, 5]:  # Test with different epoch counts
        print(f"\n=== TESTING {num_epochs} EPOCHS ===")

        with initialize(version_base=None, config_path="../config"):
            cfg = compose(
                config_name="default.yaml",
                overrides=[
                    f"model={model_type}",
                    f"model.fitness_predictor_architecture={architecture_type}",
                    f"model.training.epochs={num_epochs}",
                ],
            )

        # Use a temporary directory for this test
        with tempfile.TemporaryDirectory() as temp_dir_path:
            temp_dir = Path(temp_dir_path)
            device = select_device(cfg)
            csv_path = Path(__file__).parent.parent / "data" / "csv" / "test.csv"
            batches_dir = temp_dir / "batches"
            batches_dir.mkdir()

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

            train_output = _run_training_and_capture_output(
                cfg, temp_dir, train_dataloader, valid_dataloader
            )

            # Check that the correct number of epochs ran
            epoch_count = train_output.count("------- EPOCH")
            assert (
                epoch_count == num_epochs
            ), f"Expected {num_epochs} epochs, found {epoch_count}"
            print(f"{num_epochs} epochs completed successfully")


def _setup_test_training_output():
    """Helper function to setup training for output format test."""
    with initialize(version_base=None, config_path="../config"):
        cfg = compose(
            config_name="default.yaml",
            overrides=[
                "model=baseline",
                "model.fitness_predictor_architecture=MLP",
                "model.training.epochs=1",
            ],
        )
    return cfg


def _validate_output_patterns(train_output):
    """Helper function to validate training output patterns."""
    expected_patterns = [
        r"Starting training loop",
        r"------- EPOCH \d+ -------",
        r"Average training elbo loss: -?\d+\.\d+",
        r"Average training reconstruction loss: -?\d+\.\d+",
        r"Average training kl loss: -?\d+\.\d+",
        r"Best epoch: \[\d+\]",
        r"epoch duration:\s+\d+\.\d+s",
        r"End of training loop",
    ]

    for pattern in expected_patterns:
        assert re.search(
            pattern, train_output
        ), f"Expected pattern not found: {pattern}"

    print("All expected output patterns found")
    print("Training output format is correct")


@pytest.mark.usefixtures("_random_seeder")
def test_training_output_format():
    """Test that training output has the expected format and information."""
    cfg = _setup_test_training_output()

    with tempfile.TemporaryDirectory() as temp_dir_path:
        temp_dir = Path(temp_dir_path)
        device = select_device(cfg)
        csv_path = Path(__file__).parent.parent / "data" / "csv" / "test.csv"
        batches_dir = temp_dir / "batches"
        batches_dir.mkdir()

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

        train_output = _run_training_and_capture_output(
            cfg, temp_dir, train_dataloader, valid_dataloader
        )

        # Validate output patterns
        _validate_output_patterns(train_output)
