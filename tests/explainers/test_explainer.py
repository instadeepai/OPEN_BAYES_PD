"""Parametric tests for Captum explainers via handle_explanation path.

This replaces the deprecated SHAP explainer test. It verifies that each
explainer runs and returns explanations of the expected shape, without
asserting exact values.
"""

# pylint: disable=W0621,R0914,R0801,W0611,W0212,duplicate-code

import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyro
import pytest
import torch
from hydra import compose, initialize
from omegaconf import OmegaConf
from pyro.infer.autoguide import AutoDiagonalNormal

from src.data_processing import dataset as dataset_mod
from src.data_processing import esm_embeddings as esm_mod
from src.data_processing.dataloader import PhageDisplayDataLoader
from src.data_processing.dataset import SelectionDataset
from src.data_processing.prepare_dataset import correct_counts
from src.explainers import captum_explainers as cap_mod
from src.explainers import model_conversion_utils as conv_utils
from src.explainers.captum_explainers import (
    CaptumExplainer,
    DeepLiftCaptumExplainer,
    DeepLiftShapCaptumExplainer,
    ExplainerConfig,
    IntegratedGradientsExplainer,
    LrpCaptumExplainer,
    SaliencyCaptumExplainer,
)
from src.models.baseline_model import PhageDisplayPoissonRegressor
from src.models.core.bayesian_cnn import DeterministicBayesianCNNCounterpart
from src.models.predict_helpers import predict_bnn
from src.utils.device import select_device
from tests.conftest import _random_seeder  # noqa: F401
from tests.conftest import setup_and_destroy_experiment_dir  # noqa: F401

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def _load_and_prepare_predict_dataset(cfg, device):
    """Load and prepare prediction and background datasets."""
    # Load prediction data
    csv_p = Path(__file__).parent.parent / "data" / "csv" / "mini_test_prediction.csv"
    print(f"Loading prediction data from: {csv_p}")
    predict_df = pd.read_csv(csv_p)
    print(f"Prediction dataframe shape: {predict_df.shape}")
    print(f"Prediction columns: {list(predict_df.columns)}")

    predict_df = correct_counts(predict_df)
    predict_dataset = SelectionDataset(predict_df, "null", "null", cfg, device, True)
    print(f"Prediction dataset created with {len(predict_dataset)} samples")

    predict_dataloader = PhageDisplayDataLoader(
        predict_dataset, cfg.model.prediction.experiment_pair_predict
    )
    print("Prediction dataloader created")

    # Load background data
    csv_p = Path(__file__).parent.parent / "data" / "csv" / "mini_test_background.csv"
    print(f"Loading background data from: {csv_p}")
    background_df = pd.read_csv(csv_p)
    print(f"Background dataframe shape: {background_df.shape}")

    background_df = correct_counts(background_df)
    background_dataset = SelectionDataset(
        background_df, "null", "null", cfg, device, True
    )
    print(f"Background dataset created with {len(background_dataset)} samples")

    background_dataloader = PhageDisplayDataLoader(
        background_dataset, cfg.model.prediction.experiment_pair_background
    )

    return predict_df, predict_dataloader, background_dataloader


def _create_and_save_minimal_model(_cfg: Any, _predict_loader, _experiment_dir: str):
    """No-op: model is created directly in fake loader, skip disk IO."""
    return None


def _monkeypatch_fast_embeddings_and_baseline(
    monkeypatch: pytest.MonkeyPatch,
    cfg: Any,
):
    """Patch ESM embedding generation and Captum baselines to avoid heavy downloads."""

    # Fast ESM: return random tensors with the expected shape
    def fake_extract_batch_embeddings(
        _model: torch.nn.Module,
        sequence_batch: list[str],
    ) -> torch.Tensor:
        batch = len(sequence_batch)
        seq_len = int(cfg.model.cnn_sequence_length)
        embed_dim = int(cfg.model.embedding_size[0])
        return torch.randn(batch, seq_len, embed_dim)

    # Baseline for Captum explainers needing it
    def fake_setup_baseline(
        _self: CaptumExplainer, batch_size: int, _sl: int, _use_padding: bool
    ) -> torch.Tensor:  # type: ignore[override]
        embed_dim = int(cfg.model.embedding_size[0])
        seq_len = int(cfg.model.cnn_sequence_length)
        return torch.randn(batch_size, embed_dim, seq_len)

    # Patch in esm module and in dataset module (symbols imported there)
    monkeypatch.setattr(
        esm_mod,
        "extract_batch_embeddings",
        fake_extract_batch_embeddings,
        raising=True,
    )
    monkeypatch.setattr(
        dataset_mod,
        "extract_batch_embeddings",
        fake_extract_batch_embeddings,
        raising=True,
    )
    monkeypatch.setattr(
        CaptumExplainer,
        "setup_baseline",
        fake_setup_baseline,
        raising=True,
    )

    # Local-only model creation: no loading; initialize guide params with dummy batch
    def fake_load_model_and_parameters(
        _cfg: Any,
        device: torch.device,
        embedding_size: Any,
    ):
        pyro.clear_param_store()
        model_local = PhageDisplayPoissonRegressor(_cfg, embedding_size)
        guide_local = AutoDiagonalNormal(model_local)
        dummy = {
            "embeddings": torch.randn(
                4, embedding_size[0], embedding_size[1], device=device
            ),
            "initial_count": torch.ones(4, 1, device=device),
            "initial_total": torch.ones(4, 1, device=device),
            "selected_count": torch.ones(4, 1, device=device),
            "selected_total": torch.ones(4, 1, device=device),
            "total_sequences_in_experiment": 4,
            "experiment_name": ["DNA_100pM_2_3"] * 4,
        }
        _ = pyro.poutine.trace(guide_local).get_trace(dummy)
        return model_local, guide_local

    monkeypatch.setattr(
        "src.models.predict_helpers.load_model_and_parameters",
        fake_load_model_and_parameters,
        raising=True,
    )


def _monkeypatch_cdr_extraction(monkeypatch: pytest.MonkeyPatch):
    """Patch CDR extraction to avoid external AbNumber dependency during tests.

    Returns deterministic dummy CDR positions for each input sequence.
    """

    def fake_extract_cdr_positions_with_abnumber(predict_dataloader):
        # Return simple [start, end] pairs for CDR1/2/3 per sequence
        # This mirrors the expected structure: List[List[int]]
        out = []
        for batch in predict_dataloader:
            num_seq = len(batch.get("sequence", []))
            for _ in range(num_seq):
                out.append([0, 1, 0, 1, 0, 1])
        return out

    # Patch where the symbol is used (imported into predict_helpers)
    monkeypatch.setattr(
        "src.models.predict_helpers.extract_cdr_positions_with_abnumber",
        fake_extract_cdr_positions_with_abnumber,
        raising=True,
    )


def _load_cfg_with_hydra(
    tmp_dir: str,
    explainer_type: str,
    extra_overrides: list[str] | None = None,
):
    """Load cfg using Hydra initialize/compose with minimal overrides."""
    with initialize(version_base=None, config_path="../config"):
        overrides = [
            "model.fitness_predictor_architecture=CNN",
            f"model.prediction.model_dir_path={tmp_dir}",
            "model.prediction.explain=true",
            "model.explainer.num_model_to_explain=0",
            f"model.explainer.type={explainer_type}",
        ]
        if extra_overrides:
            overrides.extend(extra_overrides)
        cfg = compose(config_name="default.yaml", overrides=overrides)
    return cfg


def _patch_conversion_and_baselines(
    monkeypatch: pytest.MonkeyPatch,
    cfg: Any,
    in_channels: int,
    seq_len: int,
):
    """Apply common monkeypatches used across unit-shape tests."""

    def fake_convert(_model, _guide, _cfg, _weights=None):  # noqa: ARG001
        cnn_cfg = conv_utils.create_cnn_config(cfg)
        return DeterministicBayesianCNNCounterpart(cnn_cfg)

    monkeypatch.setattr(
        cap_mod,
        "from_pyro_to_pytorch_baseline_cnn_scorer",
        fake_convert,
        raising=True,
    )

    def fake_setup_baseline(_self, bsz, _sl, _use_padding):  # noqa: ARG001
        return torch.randn(bsz, in_channels, seq_len)

    monkeypatch.setattr(
        CaptumExplainer, "setup_baseline", fake_setup_baseline, raising=True
    )

    monkeypatch.setattr(
        cap_mod,
        "sample_weights_from_guide",
        lambda guide, use_mean_model: None,  # noqa: ARG001
        raising=True,
    )


def _fake_aggregate(all_relevance_scores):
    """Aggregation helper extracted to reduce test complexity."""
    first = all_relevance_scores[0]
    result = {"mean": []}
    if isinstance(first, list):
        num_data = len(first)
        for i in range(num_data):
            stacked = np.stack(
                [model_scores[i] for model_scores in all_relevance_scores]
            )
            result["mean"].append(np.mean(stacked, axis=0))
    else:
        stacked = np.stack(all_relevance_scores)
        result["mean"].append(np.mean(stacked, axis=0))
    return result


def _attach_aggregate_patch(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        CaptumExplainer,
        "_aggregate_relevance_scores",
        lambda self, x, *args, **kwargs: _fake_aggregate(x),
        raising=True,
    )


def _build_data_to_explain(batch_size: int, in_channels: int, seq_len: int):
    return [
        {"embeddings": torch.randn(1, in_channels, seq_len)} for _ in range(batch_size)
    ]


def _make_explainer(explainer_key: str, cfg: Any, needs_baseline: bool):
    config = ExplainerConfig(
        baseline_size=(1 if needs_baseline else None),
        model=object(),
        guide=None,
        cfg=cfg,
    )
    mapping = {
        "deeplift": DeepLiftCaptumExplainer,
        "deeplift_shap": DeepLiftShapCaptumExplainer,
        "integrated_gradients": IntegratedGradientsExplainer,
        "saliency": SaliencyCaptumExplainer,
    }
    return mapping[explainer_key](config=config)


def _assert_explanations_shape(out, batch_size: int, seq_len: int):
    out = (
        out[0]["mean"],
        out[0]["union"] if "union" in out[0] else [],
        out[0]["intersection"] if "intersection" in out[0] else [],
        out[0]["uai_plus"] if "uai_plus" in out[0] else [],
    )
    assert isinstance(out, tuple)
    for batch in out:
        assert len(batch) == batch_size or len(batch) == 0
        for vec in batch:
            if isinstance(vec, torch.Tensor):
                vec = vec.detach().cpu().numpy()
            assert isinstance(vec, np.ndarray)
            assert vec.ndim == 1
            assert vec.shape[0] == seq_len


@pytest.mark.parametrize(
    ("explainer_type", "extra_overrides"),
    [
        ("lrp_captum", []),
        ("deeplift", []),
        ("deeplift_shap", ["model.explainer.deeplift_shap.num_baselines=2"]),
        (
            "integrated_gradients",
            ["model.explainer.integrated_gradients.steps=4"],
        ),
        ("gradcam_captum", ["model.explainer.grad_cam.target_cnn_layer=0"]),
    ],
)
@pytest.mark.usefixtures("_random_seeder")
def test_handle_explanation_runs_and_shapes(
    setup_and_destroy_experiment_dir: str,  # noqa: F811
    explainer_type: str,
    extra_overrides: list[str],
    monkeypatch: pytest.MonkeyPatch,
):
    """Runs each explainer end-to-end and checks output shapes only."""

    tmp_dir = setup_and_destroy_experiment_dir

    cfg = _load_cfg_with_hydra(tmp_dir, explainer_type, extra_overrides)
    if extra_overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(extra_overrides))

    device = select_device(cfg)

    # Fast paths for embeddings and baselines
    _monkeypatch_fast_embeddings_and_baseline(monkeypatch, cfg)

    # Data
    predict_df, predict_loader, background_loader = _load_and_prepare_predict_dataset(
        cfg, device
    )

    # Minimal saved model and params
    _create_and_save_minimal_model(
        cfg, predict_loader, cfg.model.prediction.model_dir_path
    )

    # Patch CDR extraction to return dummy positions and decouple AbNumber
    _monkeypatch_cdr_extraction(monkeypatch)

    # Predict with explanations
    output = predict_bnn(
        to_predict=predict_loader,
        cfg=cfg,
        device=device,
        embedding_size=cfg.model.embedding_size,
        background=background_loader,
        cdrs=None,
        save_images=False,
    )

    key = f"{explainer_type}_values"
    assert key in output, f"Missing explanation key: {key}"

    values = output[key]
    assert isinstance(values, list), "Explanations must be a list per data point"
    assert len(values) == len(predict_df), "One explanation per input row"

    # Each item must be a 1D tensor/array with length == sequence length
    expected_len = int(cfg.model.cnn_sequence_length)
    for idx, arr in enumerate(values):
        if isinstance(arr, torch.Tensor):
            arr_np = arr.detach().cpu().numpy()
        elif isinstance(arr, np.ndarray):
            arr_np = arr
        else:
            raise AssertionError(
                f"Explanation {idx} must be np.ndarray or torch.Tensor,"
                f" got {type(arr)}"
            )

        assert (
            arr_np.ndim == 1
        ), f"Explanation {idx} must be 1D, got shape {arr_np.shape}"
        assert arr_np.shape[0] == expected_len, (
            f"Explanation {idx} length must be {expected_len},"
            f" got {arr_np.shape[0]}"
        )
        assert np.all(np.isfinite(arr_np)), f"Explanation {idx} contains NaN or Inf"


@pytest.mark.parametrize(
    ("explainer_key", "needs_baseline"),
    [
        ("deeplift", True),
        ("deeplift_shap", True),
        ("integrated_gradients", True),
    ],
)
def test_captum_explainers_unit_shapes(
    monkeypatch: pytest.MonkeyPatch, explainer_key: str, needs_baseline: bool
):
    """Test the shapes of the explanations for the Captum explainers."""
    # Settings
    batch_size = 3

    # cfg via Hydra
    cfg = _load_cfg_with_hydra("/tmp", "lrp_captum", [])
    in_channels = int(cfg.model.fitness_predictor_cnn.list_of_channels[0])
    seq_len = int(cfg.model.cnn_sequence_length)
    _patch_conversion_and_baselines(monkeypatch, cfg, in_channels, seq_len)
    _attach_aggregate_patch(monkeypatch)

    data_to_explain = _build_data_to_explain(batch_size, in_channels, seq_len)
    explainer = _make_explainer(explainer_key, cfg, needs_baseline)
    explainer.setup_model(weights=None)

    preds = [0.5] * batch_size
    stds = [0.1] * batch_size
    cdrs = [[0, 1, 0, 1, 0, 1] for _ in range(batch_size)]
    out = explainer.explain(
        data_to_explain=data_to_explain,
        prediction=preds,
        prediction_confidence=stds,
        cdrs=cdrs,
        temp_address="/tmp",
        save_images=False,
        calculate_metrics=False,
        num_model_to_explain=1,
    )

    _assert_explanations_shape(out, batch_size, seq_len)


def test_prepare_explanation_kwargs_adds_baseline(monkeypatch: pytest.MonkeyPatch):
    """Test that the baseline is added to the explanation kwargs."""
    # cfg via Hydra
    cfg = _load_cfg_with_hydra("/tmp", "lrp_captum", [])
    in_channels = int(cfg.model.fitness_predictor_cnn.list_of_channels[0])
    seq_len = int(cfg.model.cnn_sequence_length)

    def fake_setup_baseline(_self, bsz, _sl, _use_padding):  # noqa: ARG001
        return torch.randn(bsz, in_channels, seq_len)

    monkeypatch.setattr(
        CaptumExplainer, "setup_baseline", fake_setup_baseline, raising=True
    )

    config = ExplainerConfig(
        baseline_size=2,
        model=None,
        guide=None,
        cfg=cfg,
    )
    expl = CaptumExplainer(config)
    # Access private on purpose: unit test of helper behavior
    kw = expl._prepare_explanation_kwargs()
    assert "baselines" in kw
    assert isinstance(kw["baselines"], torch.Tensor)
    assert kw["baselines"].shape == (2, in_channels, seq_len)


def test_is_num_model_zero_sets_flag():
    """Test that the flag is set correctly when the number of models is zero."""
    # cfg from ../config/default.yaml
    cfg = _load_cfg_with_hydra(None, "lrp_captum", [])

    expl = CaptumExplainer(ExplainerConfig(cfg=cfg))
    # Access private on purpose: unit test of helper behavior
    out = expl._is_num_model_zero(0)
    assert out == 1
    assert expl.state.use_mean_model is True


def test_lrp_setup_rules_assigns_rule(monkeypatch: pytest.MonkeyPatch):
    """Test that the rule is assigned correctly to the LRP explainer."""
    cfg = _load_cfg_with_hydra("/tmp", "lrp_captum", [])

    def fake_convert(_model, _guide, _cfg, _weights=None):
        cnn_cfg = conv_utils.create_cnn_config(cfg)
        return DeterministicBayesianCNNCounterpart(cnn_cfg)

    # Patch where used
    monkeypatch.setattr(
        cap_mod,
        "from_pyro_to_pytorch_baseline_cnn_scorer",
        fake_convert,
        raising=True,
    )

    expl = LrpCaptumExplainer(ExplainerConfig(cfg=cfg))
    expl.setup_model(weights=None)

    # conv layer should have a 'rule' attribute attached by _setup_lrp_rules
    for layer in expl.state.affinity_predictor.modules():
        if isinstance(layer, torch.nn.Conv1d):
            assert hasattr(layer, "rule")
            break
