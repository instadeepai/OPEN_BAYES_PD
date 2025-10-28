# pylint: disable=W0621,W0611,W0104
import os
import sys
from pathlib import Path

import pytest
from hydra import compose, initialize

from src.data_processing.prepare_dataset import prepare_dataset_for_training
from src.utils.device import select_device
from tests.conftest import _random_seeder  # noqa: F401
from tests.conftest import setup_and_destroy_experiment_dir  # noqa: F401

# Add tests directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


@pytest.mark.usefixtures("_random_seeder")
def test_datasets_are_prepared_correctly(
    setup_and_destroy_experiment_dir,  # noqa: F811
):
    """test that the dataset are prepared correctly"""
    with initialize(version_base=None, config_path="../config"):
        cfg = compose(config_name="default.yaml")

    device = select_device(cfg)
    csv_path = Path(__file__).parent.parent / "data" / "csv" / "test.csv"
    experiment_dir = setup_and_destroy_experiment_dir
    batches_dir = experiment_dir / "batches"
    batches_dir.mkdir()

    (
        train_dataset,
        valid_dataset,
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

    # assert correct dataset length
    assert len(train_dataset) == 8
    assert len(valid_dataset) == 5

    # assert expected embedding dimensions
    for i, batch in enumerate(train_dataloader):
        if i == 0:
            assert list(batch["embeddings"].shape) == [8, 320]
        else:
            assert list(batch["embeddings"].shape) == [8, 320]
    for batch in valid_dataloader:
        assert list(batch["embeddings"].shape) == [5, 320]
