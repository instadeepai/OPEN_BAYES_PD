# pylint: disable=W0621,W0611,W0104,C0103,R0915,R0914,R0903,C0415
"""Tests for prepare_dataset_for_training and dataloaders.

Embeddings are faked locally to speed up tests; we verify shapes/keys/lengths.
"""

from pathlib import Path
from typing import Iterator
from unittest.mock import patch

import pytest
import torch
from hydra import compose, initialize

from src.data_processing.dataloader import LazyPhageDisplayDataLoader
from src.data_processing.prepare_dataset import (
    create_dataloader,
    prepare_dataset_for_training,
)
from src.utils.device import select_device
from tests.conftest import _random_seeder  # noqa: F401
from tests.conftest import setup_and_destroy_experiment_dir  # noqa: F401


class _FakeESM(torch.nn.Module):
    """Fake ESM model."""

    def __init__(self, seq_max_length: int):
        super().__init__()
        self.seq_max_length = seq_max_length

    def forward(self, sequence_batch: list[str]) -> torch.Tensor:
        """Forward pass."""
        batch = len(sequence_batch)
        # small embedding dim for speed
        return torch.zeros(batch, self.seq_max_length, 8)


@pytest.fixture
def _fake_esm():
    """Patch ESM model."""
    with patch(
        "src.data_processing.esm_embeddings.load_esm_model",
        lambda model_name, seq_max_length, arch: _FakeESM(seq_max_length),
    ):
        yield


@pytest.mark.usefixtures(
    "_random_seeder",
    "_fake_esm",
    "setup_and_destroy_experiment_dir",
)
def test_prepare_dataset_for_training_end_to_end():
    """Test prepare dataset for training end to end."""
    with initialize(version_base=None, config_path="../config"):
        cfg = compose(config_name="default.yaml")

    device = select_device(cfg)
    base = Path(__file__).parent.parent
    csv_path = base / "data" / "csv" / "test.csv"
    batches_dir = base / "tmp_dir_experiment" / "batches"
    batches_dir.mkdir(parents=True, exist_ok=True)

    (
        train_dataset,
        valid_dataset,
        train_dataloader,
        _valid_dataloader,
        _test_dataloader,
        kd_values,
        _embedding_size,
    ) = prepare_dataset_for_training(
        csv_path=csv_path,
        cfg=cfg,
        batches_path=batches_dir,
        device=device,
        test_data_path=None,
    )

    # dataset lengths
    assert len(train_dataset) > 0
    assert len(valid_dataset) > 0

    # dataloader iteration: keys and shapes present
    first_batch = next(iter(train_dataloader))
    for key in [
        "initial_count",
        "selected_count",
        "initial_total",
        "selected_total",
        "sequence",
        "experiment_name",
        "embeddings",
        "selectivity",
        "total_sequences_in_experiment",
    ]:
        assert key in first_batch

    assert first_batch["embeddings"].ndim in (2, 3)
    assert first_batch["initial_count"].shape[1] == 1

    # kd values none for this path
    assert kd_values is None


@pytest.mark.usefixtures(
    "_random_seeder",
    "_fake_esm",
    "setup_and_destroy_experiment_dir",
)
def test_create_dataloader_lazy_and_normal():
    """Test create dataloader lazy and normal."""
    with initialize(version_base=None, config_path="../config"):
        cfg = compose(config_name="default.yaml")

    # Test both normal and lazy dataloaders
    for dataloader_type in ["normal", "lazy"]:
        cfg.preprocessing.dataloader_type = dataloader_type
        device = select_device(cfg)

        base = Path(__file__).parent.parent
        csv_path = base / "data" / "csv" / "test.csv"
        batches_dir = base / f"tmp_dir_experiment/batches_{dataloader_type}"
        batches_dir.mkdir(parents=True, exist_ok=True)

        # Create datasets directly to test create_dataloader path
        (
            train_dataset,
            valid_dataset,
            _train_dataloader,
            _valid_dataloader,
            _test_dataloader,
            _kd_values,
            _embedding_size,
        ) = prepare_dataset_for_training(
            csv_path=csv_path,
            cfg=cfg,
            batches_path=batches_dir,
            device=device,
            test_data_path=None,
        )

        # Replace with create_dataloader to explicitly exercise that function
        train_dataloader2, valid_dataloader2, _ = create_dataloader(
            cfg, train_dataset, valid_dataset, None
        )

        # Iterate one batch and validate shape consistency
        def _peek(it: Iterator):
            """Peek at the next item in the iterator."""
            return next(iter(it))

        b1 = _peek(train_dataloader2)
        b2 = _peek(valid_dataloader2)

        assert "experiment_name" in b1
        assert "embeddings" in b1
        assert "experiment_name" in b2
        assert "embeddings" in b2

        if dataloader_type == "lazy":
            assert isinstance(train_dataloader2.dataset, LazyPhageDisplayDataLoader)
