# pylint: disable=W0621,W0611,W0104,C0103,R0915,R0914,R0903,C0415
"""Tests for src/explainers/plotting_utils.py"""

from typing import List
from unittest.mock import patch

import numpy as np
import pytest


@pytest.fixture(autouse=True)
def _use_agg_backend():
    """Force matplotlib non-interactive backend for tests."""
    import matplotlib

    matplotlib.use("Agg")


def _dummy_relevance_maps(n: int, length: int) -> List[np.ndarray]:
    """Generate dummy relevance maps."""
    rng = np.random.default_rng(0)
    return [rng.normal(0, 1, size=(length,)) for _ in range(n)]


def _dummy_masks(n: int, length: int) -> List[np.ndarray]:
    """Generate dummy masks."""
    masks = []
    for i in range(n):
        mask = np.zeros(length)
        start = (i * 3) % (length // 2)
        end = min(start + 5, length - 1)
        mask[start : end + 1] = 1
        masks.append(mask)
    return masks


def test_create_cdr_ground_truth_masks_basic():
    """Test create_cdr_ground_truth_masks basic functionality."""
    from src.explainers.plotting_utils import _create_cdr_ground_truth_masks

    cdrs = [[0, 4, 10, 12], [2, 3]]
    masks = _create_cdr_ground_truth_masks(cdrs, sequence_length=20)
    assert len(masks) == 2
    assert masks[0].sum() == 8  # spans 0-4 and 10-12
    assert masks[1].sum() == 2  # 2..3 inclusive => 2


def test_calculate_auc_score_happy_path():
    """Test calculate_auc_score happy path."""
    from src.explainers.plotting_utils import _calculate_auc_score

    rel = _dummy_relevance_maps(3, 10)
    masks = _dummy_masks(3, 10)
    auc = _calculate_auc_score(rel, masks)
    assert 0.0 <= auc <= 1.0


def test_calculate_mass_accuracy_happy_path():
    """Test calculate_mass_accuracy happy path."""
    from src.explainers.plotting_utils import _calculate_mass_accuracy

    rel = _dummy_relevance_maps(3, 10)
    masks = _dummy_masks(3, 10)
    ma = _calculate_mass_accuracy(rel, masks)
    assert 0.0 <= ma <= 1.0


def test_minmax_normalize_relevance_symmetry():
    """Test minmax_normalize_relevance symmetry."""
    from src.explainers.plotting_utils import minmax_normalize_relevance

    arr = np.array([[-2.0, 0.0, 4.0]])
    norm = minmax_normalize_relevance(arr)
    # positive max maps to 1, negative min maps to -1
    assert np.isclose(norm[0, 2], 1.0)
    assert np.isclose(norm[0, 0], -1.0)


def test_process_relevance_for_plotting_shapes():
    """Test process_relevance_for_plotting shapes."""
    from src.explainers.plotting_utils import process_relevance_for_plotting

    one_d = np.array([1.0, 2.0, 3.0])
    two_d = process_relevance_for_plotting(one_d)
    assert two_d.ndim == 2
    assert two_d.shape[0] == 1


@patch("matplotlib.pyplot.savefig")
@patch("matplotlib.pyplot.show")
def test_plot_statistical_analysis_single_model(mock_show, mock_savefig, tmp_path):
    """Test plot_statistical_analysis single model."""
    from src.explainers.plotting_utils import create_statistical_analysis

    # one-model case -> only mean present/utilized
    all_rel = {"mean": [np.random.rand(50)]}
    cdrs = [[2, 6, 10, 12]]
    create_statistical_analysis(
        all_rel,
        cdrs,
        str(tmp_path),
        explainer_name="testexp",
        more_than_one_model=False,
        sequence_length=50,
    )
    mock_savefig.assert_called_once()
    mock_show.assert_called_once()


@patch("matplotlib.pyplot.savefig")
@patch("matplotlib.pyplot.show")
def test_plot_statistical_analysis_multi_model(mock_show, mock_savefig, tmp_path):
    """Test plot_statistical_analysis multi model."""
    from src.explainers.plotting_utils import create_statistical_analysis

    # multi-model case -> mean/union/intersection/uai_plus present
    maps = [np.random.rand(50) for _ in range(3)]
    all_rel = {
        "mean": maps,
        "union": maps,
        "intersection": maps,
        "uai_plus": maps,
    }
    cdrs = [[2, 6, 10, 12]] * 3
    create_statistical_analysis(
        all_rel,
        cdrs,
        str(tmp_path),
        explainer_name="testexp",
        more_than_one_model=True,
        sequence_length=50,
    )
    mock_savefig.assert_called_once()
    mock_show.assert_called_once()


@patch("matplotlib.pyplot.savefig")
def test_create_individual_plot_saves(mock_savefig, tmp_path):
    """Test create_individual_plot saves."""
    from src.explainers.plotting_utils import create_individual_plot

    plot_data = (
        0.8,  # pred
        0.1,  # pred_error
        "expA",  # name
        "ACDEFGHIKLMNPQRSTVWY",  # seq length 20
        0.75,  # pred_deter
        np.random.randn(20),  # relevance
    )
    create_individual_plot(
        x_idx=0,
        plot_data=plot_data,
        cdrs=[[2, 4, 10, 12]],
        temp_address=str(tmp_path),
        explainer_name="mean",
        auc_ma_per_agg={"mean": {"auc_roc": [0.9], "mass_accuracy": [0.8]}},
        sequence_length=20,
    )
    mock_savefig.assert_called_once()


@patch("matplotlib.pyplot.savefig")
def test_create_individual_plot_multi_saves(mock_savefig, tmp_path):
    """Test create_individual_plot_multi saves."""
    from src.explainers.plotting_utils import create_individual_plot_multi

    plot_data = (
        0.8,
        0.1,
        "expA",
        "ACDEFGHIKLMNPQRSTVWY",
        0.75,
        np.random.randn(20),
        np.random.randn(20),
        np.random.randn(20),
        np.random.randn(20),
    )
    create_individual_plot_multi(
        x_idx=0,
        plot_data=plot_data,
        cdrs=[[2, 4, 10, 12]],
        temp_address=str(tmp_path),
        explainer_name="gradcam",
        auc_ma_per_agg={
            "mean": {"auc_roc": [0.9], "mass_accuracy": [0.8]},
            "union": {"auc_roc": [0.85], "mass_accuracy": [0.75]},
            "intersection": {"auc_roc": [0.83], "mass_accuracy": [0.73]},
            "uai_plus": {"auc_roc": [0.81], "mass_accuracy": [0.71]},
        },
        sequence_length=20,
    )
    mock_savefig.assert_called_once()
