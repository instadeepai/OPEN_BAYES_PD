# pylint: disable=W0621,W0611,W0104,C0103,R0915,R0914,R0903,C0415
"""Minimal tests for scripts - just run main with mocks"""

import os
import sys
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import torch
from omegaconf import OmegaConf

from tests.conftest import _random_seeder  # noqa: F401

# Add tests directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


class TestScripts:
    """Test both train and predict scripts"""

    @pytest.mark.usefixtures("_random_seeder")
    @patch("scripts.train.select_device")
    @patch("scripts.train.prepare_dataset_for_training")
    @patch("scripts.train.train_bnn")
    @patch("scripts.train.neptune.init_run")
    @patch("scripts.train.neptune_add_info")
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    def test_train_main_runs(
        self,
        mock_neptune_add_info,
        mock_neptune,
        mock_train_bnn,
        mock_prepare,
        mock_device,
        tmp_path,
    ):
        """Test that train script main function runs"""
        # Setup mocks with dummy values
        mock_device.return_value = "cpu"

        mock_neptune_run = Mock()
        # Configure Neptune mock to support subscriptable access
        mock_sys_id = Mock()
        mock_sys_id.fetch.return_value = "test_run_id_123"
        mock_neptune_run.__getitem__ = Mock(return_value=mock_sys_id)
        mock_neptune.return_value = mock_neptune_run

        # Create a mock dataloader with proper data dictionary
        mock_dataloader = Mock()
        mock_dataloader.dataset = Mock()
        mock_data = {
            "initial_count": torch.tensor([1.0, 2.0]),
            "initial_total": torch.tensor([3.0]),
            "selectivity": torch.tensor([0.5, 0.7]),
        }
        mock_dataloader.dataset.data = mock_data
        mock_prepare.return_value = (
            mock_dataloader,
            Mock(),
            Mock(),
            Mock(),
            Mock(),
            Mock(),
            128,
        )

        # Create simple mock config
        cfg = OmegaConf.create({})
        cfg.model = {}
        cfg.model.training = {}
        cfg.model.prediction = {}
        cfg.model.splits = {}
        cfg.paths = {}
        cfg.paths.data_dir = str(tmp_path)
        cfg.paths.results_dir = str(tmp_path)
        cfg.model.training.relative_csv_file_location = "test.csv"
        cfg.model.training.experiment_name = "test_exp"
        cfg.model.training.test_data_dir = None
        cfg.model.training.relative_batches_location = None
        cfg.model.training.relative_model_checkpoint_location = None
        cfg.model.training.number_of_particles = 10
        cfg.model.training.epochs = 1
        cfg.neptune_project = "test/project"
        cfg.use_neptune = True
        cfg.model.phage_initial_population_size = 1000.0
        cfg.model.sequencing_depth = 10000.0
        # Add splits configuration
        cfg.model.splits.training_pair = ["test_training_pair"]
        cfg.model.splits.validation_pair = ["test_validation_pair"]
        # Add tags configuration
        cfg.model.tags = ["test_tag"]

        # Add fitness predictor architecture
        cfg.model.fitness_predictor_mlp = {}
        cfg.model.fitness_predictor_architecture = "MLP"
        cfg.model.fitness_predictor_mlp.layers_size = [128, 64, 1]

        from scripts.train import main as train_main

        # Mock calculate_null_model_correlation to use the data directly
        with patch("scripts.train.calculate_null_model_correlation") as mock_calc:
            mock_calc.return_value = 0.5
            train_main(cfg)

        # Verify it ran
        mock_train_bnn.assert_called_once()

        # Check train_bnn was called with correct arguments
        args, kwargs = mock_train_bnn.call_args
        assert (
            len(args) + len(kwargs) == 10
        ), f"train_bnn called with {len(args) + len(kwargs)} args, expected 10"

        # Check neptune_add_info was called when use_neptune is True
        mock_neptune_add_info.assert_called_once()

    @pytest.mark.usefixtures("_random_seeder")
    @patch("scripts.predict.select_device")
    @patch("scripts.predict.load_csv_local")
    @patch("scripts.predict.adding_columns_to_df_for_pd_prediction")
    @patch("scripts.predict.correct_counts")
    @patch("scripts.predict.SelectionDataset")
    @patch("scripts.predict.PhageDisplayDataLoader")
    @patch("scripts.predict.predict_bnn")
    @patch("scripts.predict._save_prediction_results")
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    def test_predict_main_runs(
        self,
        mock_save_results,
        mock_predict_bnn,
        mock_dataloader,
        mock_dataset,
        mock_correct,
        mock_adding,
        mock_load_csv,
        mock_device,
        tmp_path,
    ):
        """Test that predict script main function runs"""
        # Setup mocks with dummy values
        mock_device.return_value = "cpu"

        # Mock result saving to avoid pandas/filesystem issues
        mock_save_results.return_value = None

        # Mock dataframe with necessary methods
        mock_df = Mock()
        mock_df.embedding_size.return_value = 128
        mock_df.columns = ["sequence", "count"]
        mock_df.__getitem__ = Mock(
            return_value=Mock(to_list=Mock(return_value=["seq1", "seq2"]))
        )
        mock_df.to_list = Mock(return_value=["seq1", "seq2"])

        mock_load_csv.return_value = mock_df
        mock_adding.return_value = mock_df
        mock_correct.return_value = mock_df

        # Mock predict output
        mock_predict_bnn.return_value = {
            "preds_proba_mean": [0.1, 0.2],
            "preds_proba_std": [0.01, 0.02],
            "preds_proba_neg_mean": [0.05, 0.1],
            "preds_proba_neg_std": [0.005, 0.01],
        }

        # Create simple mock config
        cfg = OmegaConf.create({})
        cfg.model = {}

        # FIX: Initialize the next level of keys before using them
        cfg.model.training = {}
        cfg.model.prediction = {}
        cfg.paths = {}
        cfg.paths.data_dir = str(tmp_path)
        cfg.paths.results_dir = str(tmp_path)
        cfg.model.training.test_data_dir = None
        cfg.model.prediction.to_predict_csv_file_location = "test.csv"
        cfg.model.prediction.experiment_name = "test_exp"
        cfg.model.prediction.experiment_pair_predict = ["test_pair"]
        cfg.model.prediction.background_csv_file_location = None
        cfg.model.prediction.explain = False
        cfg.model.prediction.use_training_cfg = False

        # Add tags configuration
        cfg.model.tags = ["test_tag"]

        from scripts.predict import main as predict_main

        predict_main(cfg)

        # Verify it ran
        mock_predict_bnn.assert_called_once()
        mock_save_results.assert_called_once()

        # Check predict_bnn was called with correct arguments
        args, kwargs = mock_predict_bnn.call_args
        assert (
            len(args) + len(kwargs) == 5
        ), f"predict_bnn called with {len(args)} args, expected 5"

        # Check that SelectionDataset and PhageDisplayDataLoader were called
        mock_dataset.assert_called()
        mock_dataloader.assert_called()
