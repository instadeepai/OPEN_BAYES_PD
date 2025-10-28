# pylint: disable=W0212,C0301,E1129,W0612,R0913,R0914,R0917
from typing import List, Optional, Union

import numpy as np
import torch
from captum.metrics import infidelity_perturb_func_decorator
from sklearn.metrics import roc_auc_score

from src.explainers.plotting_utils import _create_cdr_ground_truth_masks


@infidelity_perturb_func_decorator(False)
def default_perturb_fn(inputs: torch.Tensor, _target: int) -> torch.Tensor:
    """Perturb embeddings by adding random noise to random positions."""
    x_copy = inputs.clone()

    # Handle different input shapes
    if len(x_copy.shape) == 3:  # (batch, channels, sequence_length)
        num_perturbations = torch.randint(5, 15, (1,)).item()
        for _ in range(num_perturbations):
            random_pos = torch.randint(0, x_copy.shape[2], (1,)).item()
            noise = torch.randn(x_copy.shape[1]) * 0.1
            x_copy[0, :, random_pos] += noise
    elif len(x_copy.shape) == 2:  # (batch, features)
        num_perturbations = torch.randint(5, 15, (1,)).item()
        for _ in range(num_perturbations):
            random_pos = torch.randint(0, x_copy.shape[1], (1,)).item()
            noise = torch.randn(1) * 0.1
            x_copy[0, random_pos] += noise
    else:
        noise = torch.randn_like(x_copy) * 0.1
        x_copy += noise

    return x_copy


def _reduce_attributions_to_relevance(attributions: torch.Tensor) -> torch.Tensor:
    """Reduce attributions to per-position relevance (B, sequence_length)."""
    rel = attributions.detach().cpu()
    if rel.dim() == 3:
        return torch.sum(rel, dim=1)
    if rel.dim() == 2:
        return rel
    return rel.view(rel.shape[0], -1)


def _safe_roc_auc(mask: np.ndarray, relevance: np.ndarray) -> float:
    try:
        if np.any(mask == 1.0) and np.any(mask == 0.0):
            return float(roc_auc_score(mask, relevance))
        return float("nan")
    except Exception:  # pylint: disable=broad-except
        return float("nan")


def _mass_accuracy(mask: np.ndarray, relevance: np.ndarray) -> float:
    total_mass = float(relevance.sum())
    if total_mass > 0.0:
        return float((relevance * mask).sum() / total_mass)
    return float("nan")


def calculate_captum_metrics(
    attributions: torch.Tensor,
    cdr_positions: Optional[Union[List[int], List[List[int]]]] = None,
    sequence_length: int = 150,
) -> dict:
    """Calculate Captum explicability metrics for attributions with low complexity.
    This function is used to calculate the metrics for a single sequence.
    It calculates some metrics that are not used in the paper, and they
    are commented out.

    Args:
        model: The model to explain.
        input_tensor: The input tensor to explain.
        attributions: The attributions to explain.
        target: The target to explain.
        baseline: The baseline to explain.
        perturbation_fn: The perturbation function to use.
        cdr_positions: The CDR positions to explain.
        sequence_length: The sequence length to explain.

    Returns:
        A dictionary containing the metrics.
        The keys are:
        - auc_roc: The AUC ROC score.
        - mass_accuracy: The mass accuracy score.
        - mean_attribution_magnitude: The mean attribution magnitude.
        - std_attribution_magnitude: The standard deviation of the attribution magnitude
        - positive_attribution_ratio: The ratio of positive attributions.
        - negative_attribution_ratio: The ratio of negative attributions.
    """
    metrics: dict = {}

    # there was other metrics, for the moment we deprecated it.

    # AUC ROC and Mass Accuracy
    relevance = _reduce_attributions_to_relevance(attributions)
    relevance_np = np.abs(relevance.numpy())
    mask = _create_cdr_ground_truth_masks(cdr_positions, sequence_length)

    auc_vals: List[float] = []
    ma_vals: List[float] = []
    for idx in range(relevance_np.shape[0]):
        rel_flat = np.abs(relevance_np[idx].reshape(-1))
        if mask is None:
            auc_vals.append(float("nan"))
            ma_vals.append(float("nan"))
            continue
        mask_flat = np.asarray(mask).reshape(-1)
        length = min(rel_flat.shape[0], mask_flat.shape[0])
        rel_flat = rel_flat[:length]
        mask_flat = mask_flat[:length]
        auc_vals.append(_safe_roc_auc(mask_flat, rel_flat))
        ma_vals.append(_mass_accuracy(mask_flat, rel_flat))

    metrics["auc_roc"] = np.asarray(auc_vals)
    metrics["mass_accuracy"] = np.asarray(ma_vals)
    return metrics


def calculate_auc_ma_for_aggregated_maps(
    relevance_maps: dict,
    cdrs: Optional[list],
    sequence_length: int = 150,
    include_multi: bool = True,
) -> dict[str, dict[str, np.ndarray]]:
    """Compute per-sequence AUC ROC and Mass Accuracy for aggregated maps."""
    results: dict[str, dict[str, np.ndarray]] = {}
    if relevance_maps is None or cdrs is None:
        return results

    masks = _create_cdr_ground_truth_masks(cdrs, sequence_length)
    if not masks:
        return results

    keys = (
        ["mean"] if not include_multi else ["mean", "union", "intersection", "uai_plus"]
    )
    for key in keys:
        if key not in relevance_maps:
            continue
        auc_vals: list[float] = []
        ma_vals: list[float] = []
        maps_list = relevance_maps[key]
        num_items = min(len(maps_list), len(masks))
        for idx in range(num_items):
            rel = np.asarray(maps_list[idx]).reshape(-1)
            rel = np.abs(rel)
            mask = np.asarray(masks[idx]).reshape(-1)
            length = min(rel.shape[0], mask.shape[0])
            rel = rel[:length]
            mask = mask[:length]
            auc_vals.append(_safe_roc_auc(mask, rel))
            ma_vals.append(_mass_accuracy(mask, rel))

        results[key] = {
            "auc_roc": np.asarray(auc_vals),
            "mass_accuracy": np.asarray(ma_vals),
        }

    return results
