# pylint: disable=R0913,R0914,R0917
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

plt.rcParams["figure.dpi"] = 200

# Standard per-heatmap size derived from Matplotlib defaults to keep consistency
DEFAULT_FIGSIZE = tuple(plt.rcParams.get("figure.figsize", [6.4, 4.8]))
SINGLE_HEATMAP_FIGSIZE = (DEFAULT_FIGSIZE[0], DEFAULT_FIGSIZE[1])


def create_statistical_analysis(
    all_relevance_scores: dict,
    cdrs: List[List[int]],
    temp_address: str,
    explainer_name: str,
    more_than_one_model: bool = False,
    sequence_length: int = 150,
) -> None:
    """
    Create statistical plots depending on the number of models.
    If more_than_one_model is True, plot all maps (mean, union,
    intersection, uai_plus) and metrics. Otherwise, plot only the mean map.
    """
    ground_truth_masks = _create_cdr_ground_truth_masks(cdrs, sequence_length)

    if more_than_one_model:
        mean_maps = []
        union_maps = []
        intersection_maps = []
        uai_plus_maps = []

        for i in range(len(all_relevance_scores["mean"])):
            mean_map = all_relevance_scores["mean"][i]
            union_map = all_relevance_scores["union"][i]
            intersection_map = all_relevance_scores["intersection"][i]
            uai_plus_map = all_relevance_scores["uai_plus"][i]

            mean_maps.append(mean_map)
            union_maps.append(union_map)
            intersection_maps.append(intersection_map)
            uai_plus_maps.append(uai_plus_map)

        avg = {
            "mean": np.mean(np.abs(mean_maps), axis=0),
            "union": np.mean(np.abs(union_maps), axis=0),
            "intersection": np.mean(np.abs(intersection_maps), axis=0),
            "uai_plus": np.mean(np.abs(uai_plus_maps), axis=0),
        }

        auc = {
            "mean": _calculate_auc_score(mean_maps, ground_truth_masks),
            "union": _calculate_auc_score(union_maps, ground_truth_masks),
            "intersection": _calculate_auc_score(intersection_maps, ground_truth_masks),
            "uai_plus": _calculate_auc_score(uai_plus_maps, ground_truth_masks),
        }
        ma = {
            "mean": _calculate_mass_accuracy(mean_maps, ground_truth_masks),
            "union": _calculate_mass_accuracy(union_maps, ground_truth_masks),
            "intersection": _calculate_mass_accuracy(
                intersection_maps, ground_truth_masks
            ),
            "uai_plus": _calculate_mass_accuracy(uai_plus_maps, ground_truth_masks),
        }

    else:
        # Case with only one model: plot only the mean map
        mean_maps = []
        for i in range(len(all_relevance_scores["mean"])):
            mean_map = all_relevance_scores["mean"][i]
            mean_maps.append(mean_map)
        avg_mean = np.mean(np.abs(mean_maps), axis=0)
        auc_mean = _calculate_auc_score(mean_maps, ground_truth_masks)
        ma_mean = _calculate_mass_accuracy(mean_maps, ground_truth_masks)
        # For the case with only one model, the metrics and maps other
        # than the mean are not calculated
        avg = {
            "mean": avg_mean,
        }
        auc = {
            "mean": auc_mean,
        }
        ma = {
            "mean": ma_mean,
        }

    _plot_statistical_analysis_new(
        avg,
        auc,
        ma,
        cdrs,
        temp_address,
        explainer_name,
        sequence_length,
    )


def _create_cdr_ground_truth_masks(
    cdrs: List[List[int]], sequence_length: int
) -> List[np.ndarray]:
    """Create binary ground truth masks where CDRs are 1, rest is 0."""
    masks = []

    if not isinstance(cdrs[0], list):
        cdrs = [cdrs]

    for cdr_positions in cdrs:
        mask = np.zeros(sequence_length)
        if cdr_positions is not None:
            for i in range(0, len(cdr_positions), 2):
                if i + 1 < len(cdr_positions):
                    start = int(cdr_positions[i])
                    end = int(cdr_positions[i + 1])
                    start = max(0, min(start, sequence_length - 1))
                    end = max(0, min(end, sequence_length - 1))
                    mask[start : end + 1] = 1
        masks.append(mask)
    return masks


def _calculate_auc_score(
    relevance_maps: List[np.ndarray], ground_truth_masks: List[np.ndarray]
) -> float:
    """Calculate average AUC ROC score across all sequences."""
    auc_scores = []
    for rel_map, gt_mask in zip(relevance_maps, ground_truth_masks):
        # Both positive and negative relevance should be considered to be in the CDR
        rel_flat = np.abs(rel_map.flatten())
        gt_flat = gt_mask.flatten()

        try:
            auc = roc_auc_score(gt_flat, rel_flat)
            auc_scores.append(auc)
        except ValueError as e:
            raise ValueError(f"Error calculating AUC score: {e}") from e

    return float(np.mean(auc_scores)) if auc_scores else 0.0


def _calculate_mass_accuracy(
    relevance_maps: List[np.ndarray], ground_truth_masks: List[np.ndarray]
) -> float:
    """Calculate Relevance Mass Accuracy: proportion of relevance mass on CDRs."""
    ma_scores = []
    for rel_map, gt_mask in zip(relevance_maps, ground_truth_masks):
        # Both positive and negative relevance should be considered to be in the CDR
        rel_flat = np.abs(rel_map.flatten())
        gt_flat = gt_mask.flatten()

        mass_on_object = np.sum(rel_flat * gt_flat)
        total_mass = np.sum(rel_flat)
        if total_mass > 0:
            ma_scores.append(mass_on_object / total_mass)

    return float(np.mean(ma_scores)) if ma_scores else 0.0


def _plot_statistical_analysis_new(
    avg: Dict[str, np.ndarray],
    auc: Dict[str, np.ndarray],
    ma: Dict[str, np.ndarray],
    cdrs: List[List[int]],
    temp_address: str,
    explainer_name: str,
    sequence_length: int,
) -> None:
    """Plot the statistical analysis with 4 plots and metrics."""
    more_than_one_model = "union" in avg.keys()
    if more_than_one_model:
        fig, axes = plt.subplots(
            2,
            2,
            figsize=(SINGLE_HEATMAP_FIGSIZE[0] * 2, SINGLE_HEATMAP_FIGSIZE[1] * 2),
        )
    else:
        fig, axs = plt.subplots(1, 1, figsize=SINGLE_HEATMAP_FIGSIZE)
        axes = [[axs]]

    avg_mean_img = minmax_normalize_relevance(
        process_relevance_for_plotting(avg["mean"])
    )
    plot_relevance_heatmap(
        axes[0][0], avg_mean_img, "Mean", vmin=0, vmax=1.0, cmap="Reds"
    )

    add_cdr_annotations(axes[0][0], cdrs)

    if more_than_one_model:
        avg_union_img = minmax_normalize_relevance(
            process_relevance_for_plotting(avg["union"])
        )
        avg_intersection_img = minmax_normalize_relevance(
            process_relevance_for_plotting(avg["intersection"])
        )
        avg_uai_plus_img = process_relevance_for_plotting(avg["uai_plus"])

        plot_relevance_heatmap(
            axes[0][1], avg_union_img, "Union (α=95)", vmin=0, vmax=1.0, cmap="Reds"
        )
        add_cdr_annotations(axes[0][1], cdrs)

        plot_relevance_heatmap(
            axes[1][0],
            avg_intersection_img,
            "Intersection (α=5)",
            vmin=0,
            vmax=1.0,
            cmap="Reds",
        )
        add_cdr_annotations(axes[1][0], cdrs)

        plot_relevance_heatmap(
            axes[1][1],
            avg_uai_plus_img,
            "UAI+ (P > ε)",
            vmin=0,
            vmax=1.0,
            cmap="Reds",
            colorbar_label="Probability",
        )
        add_cdr_annotations(axes[1][1], cdrs)

        for ax in axes.flatten():
            ax.set_xlim(0, sequence_length)
            ax.set_xticks([0, sequence_length // 2, sequence_length - 1])
            ax.set_xticklabels([0, sequence_length // 2, sequence_length])
            ax.tick_params(axis="x", labelsize=12)

    # Build a compact metrics table placed below the plots
    if more_than_one_model:
        col_labels = ["Mean", "Union (α=95)", "Intersection (α=5)", "UAI+"]
        cell_text = [
            [
                f"{auc['mean']:.3f}",
                f"{auc['union']:.3f}",
                f"{auc['intersection']:.3f}",
                f"{auc['uai_plus']:.3f}",
            ],
            [
                f"{ma['mean']:.3f}",
                f"{ma['union']:.3f}",
                f"{ma['intersection']:.3f}",
                f"{ma['uai_plus']:.3f}",
            ],
        ]
        bottom_margin = 0.28
        table_height = 0.16
    else:
        col_labels = ["Mean"]
        cell_text = [[f"{auc['mean']:.3f}"], [f"{ma['mean']:.3f}"]]
        bottom_margin = 0.22
        table_height = 0.12

    row_labels = ["AUC ROC", "Mass Accuracy"]

    # Use subplots_adjust only; do not call tight_layout when colorbars are present
    plt.subplots_adjust(bottom=bottom_margin)

    # Add a dedicated axis for the table, spanning the figure width
    ax_table = fig.add_axes([0.08, 0.04, 0.84, table_height])
    ax_table.axis("off")
    table = ax_table.table(
        cellText=cell_text,
        rowLabels=row_labels,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
        bbox=[0.0, 0.0, 1.0, 1.0],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.2)
    plt.savefig(
        f"{temp_address}/{explainer_name}_statistical_analysis.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()
    plt.close()


def plot_captum_values_cnn_baseline(
    data_to_explain: DataLoader,
    prediction_determinist: List[float],
    prediction: List[float],
    prediction_confidence: List[float],
    relevance_values,  # dict with mean/union/intersection/uai_plus or List[np.ndarray]
    cdrs: List[List[int]],
    temp_address: str,
    explainer_name: str,
    more_than_one_model: bool = False,
    auc_ma_per_agg: Optional[dict] = None,
    sequence_length: int = 150,
):
    """Plot Captum values for CNN baseline model.

    Handles both dict format (mean/union/intersection/uai_plus) and legacy list.
    """
    data_info = extract_data_info(data_to_explain)
    total_iterator = len(prediction)

    if more_than_one_model:
        iterator = prepare_plot_data_multi(
            prediction,
            prediction_confidence,
            data_info,
            prediction_determinist,
            relevance_values,
        )
        plot_function = create_individual_plot_multi
    else:
        iterator = prepare_plot_data(
            prediction,
            prediction_confidence,
            data_info,
            prediction_determinist,
            relevance_values["mean"],
        )
        plot_function = create_individual_plot
    
    print(iterator)

    for x_idx, plot_data in tqdm(enumerate(iterator), total=total_iterator, desc="Plotting explainability heatmaps"):
        plot_function(x_idx, plot_data, cdrs, temp_address, explainer_name, auc_ma_per_agg, sequence_length)


def extract_data_info(data_to_explain: DataLoader) -> dict:
    """Extract experiment names, sequences and input length from dataloader."""
    data_to_explain_round = []
    data_to_explain_sequence = []
    initial_size = None

    for data in data_to_explain:
        data_to_explain_round.extend(data["experiment_name"])
        data_to_explain_sequence.extend(data["sequence"])
        initial_size = data["embeddings"].shape[2]

    return {
        "experiment_names": data_to_explain_round,
        "sequences": data_to_explain_sequence,
        "initial_size": initial_size,
    }


def prepare_plot_data(
    prediction: List[float],
    prediction_confidence: List[float],
    data_info: dict,
    prediction_determinist: List[float],
    relevance_values: List[np.ndarray],
) -> zip:
    """Prepare zipped data iterator for single-plot mode."""
    return zip(
        prediction,
        prediction_confidence,
        data_info["experiment_names"],
        data_info["sequences"],
        prediction_determinist,
        relevance_values,
    )


def prepare_plot_data_multi(
    prediction: List[float],
    prediction_confidence: List[float],
    data_info: dict,
    prediction_determinist: List[float],
    relevance_values_dict: dict,
) -> zip:
    """Prepare zipped data for multi-plot: mean, union, intersection, uai_plus."""
    return zip(
        prediction,
        prediction_confidence,
        data_info["experiment_names"],
        data_info["sequences"],
        prediction_determinist,
        relevance_values_dict["mean"],
        relevance_values_dict["union"],
        relevance_values_dict["intersection"],
        relevance_values_dict["uai_plus"],
    )


def create_individual_plot(
    x_idx: int,
    plot_data: tuple,
    cdrs: List[List[int]],
    temp_address: str,
    explainer_name: str,
    auc_ma_per_agg: Optional[dict] = None,
    sequence_length: int = 150,
) -> None:
    """Create individual plot for a single sequence."""
    pred, pred_error, _name, seq, pred_deter, relevance = plot_data

    fig, ax = plt.subplots(
        1,
        1,
        figsize=SINGLE_HEATMAP_FIGSIZE,
        constrained_layout=True,
    )
    title_str = prepare_title_parts(
        pred,
        pred_error,
        pred_deter,
    )
    fig.suptitle(title_str, fontsize=16, fontweight="bold")

    relevance = process_relevance_for_plotting(relevance)
    plot_relevance_heatmap(
        ax,
        relevance,
        explainer_name,
        cmap="coolwarm",
    )

    add_cdr_annotations(ax, cdrs, x_idx)
    configure_plot_appearance(ax, seq, explainer_name, sequence_length)

    # Add a small legend-like metrics box (AUC/MA) for the mean aggregation
    add_metrics_box(ax, auc_ma_per_agg, "mean", x_idx)

    # Use constrained layout; no manual tight_layout adjustments needed
    plt.savefig(
        f"{temp_address}/{explainer_name}_explanation_{x_idx}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def create_individual_plot_multi(
    x_idx: int,
    plot_data: tuple,
    cdrs: List[List[int]],
    temp_address: str,
    explainer_name: str,
    auc_ma_per_agg: Optional[dict] = None,
    sequence_length: int = 150,
) -> None:
    """Create plot with 4 subplots: mean, union, intersection, UAI+."""
    (
        pred,
        pred_error,
        _name,
        seq,
        pred_deter,
        rel_mean,
        rel_union,
        rel_intersection,
        rel_uai_plus,
    ) = plot_data

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(SINGLE_HEATMAP_FIGSIZE[0] * 2, SINGLE_HEATMAP_FIGSIZE[1] * 2),
        constrained_layout=True,
    )

    title_str = prepare_title_parts(
        pred,
        pred_error,
        pred_deter,
    )
    fig.suptitle(title_str, fontsize=16, fontweight="bold")

    rel_mean = process_relevance_for_plotting(rel_mean)
    rel_union = process_relevance_for_plotting(rel_union)
    rel_intersection = process_relevance_for_plotting(rel_intersection)
    rel_uai_plus = process_relevance_for_plotting(rel_uai_plus)

    rel_mean_norm = minmax_normalize_relevance(rel_mean)
    rel_union_norm = minmax_normalize_relevance(rel_union)
    rel_intersection_norm = minmax_normalize_relevance(rel_intersection)

    plot_relevance_heatmap(axes[0][0], rel_mean_norm, "Mean")
    add_cdr_annotations(axes[0][0], cdrs, x_idx)
    configure_plot_appearance_multi(axes[0][0], seq, sequence_length)

    plot_relevance_heatmap(axes[0][1], rel_union_norm, "Union (α=95)")
    add_cdr_annotations(axes[0][1], cdrs, x_idx)
    configure_plot_appearance_multi(axes[0][1], seq, sequence_length)

    plot_relevance_heatmap(axes[1][0], rel_intersection_norm, "Intersection (α=5)")
    add_cdr_annotations(axes[1][0], cdrs, x_idx)
    configure_plot_appearance_multi(axes[1][0], seq, sequence_length)

    plot_relevance_heatmap(
        axes[1][1],
        rel_uai_plus,
        "UAI+ (P > ε)",
        vmin=0,
        vmax=1.0,
        cmap="Reds",
        colorbar_label="Probability",
    )
    add_cdr_annotations(axes[1][1], cdrs, x_idx)
    configure_plot_appearance_multi(axes[1][1], seq, sequence_length)

    # Add small legend-like metrics boxes (AUC/MA) for each subplot
    add_metrics_box(axes[0][0], auc_ma_per_agg, "mean", x_idx)
    add_metrics_box(axes[0][1], auc_ma_per_agg, "union", x_idx)
    add_metrics_box(axes[1][0], auc_ma_per_agg, "intersection", x_idx)
    add_metrics_box(axes[1][1], auc_ma_per_agg, "uai_plus", x_idx)

    # Use constrained layout; no manual tight_layout adjustments needed
    plt.savefig(
        f"{temp_address}/{explainer_name}_explanation_{x_idx}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def prepare_title_parts(
    pred: float,
    pred_error: float,
    pred_deter: float,
) -> str:
    """Return a nicely formatted suptitle string with binding probabilities."""
    return (
        f"Bayesian prediction: {pred:.3f} ± {pred_error:.3f} | "
        f"Deterministic mean prediction: {pred_deter:.3f}"
    )


def add_metrics_box(
    axs: plt.Axes,
    auc_ma_per_agg: Optional[dict],
    agg_key: str,
    x_idx: int,
) -> None:
    """Add a small legend-like box with AUC and MA for the given subplot."""
    if not auc_ma_per_agg or agg_key not in auc_ma_per_agg:
        return
    auc_vals = auc_ma_per_agg[agg_key].get("auc_roc")
    ma_vals = auc_ma_per_agg[agg_key].get("mass_accuracy")
    lines: List[str] = []
    if auc_vals is not None and len(auc_vals) > x_idx and np.isfinite(auc_vals[x_idx]):
        lines.append(f"AUC: {float(auc_vals[x_idx]):.3f}")
    if ma_vals is not None and len(ma_vals) > x_idx and np.isfinite(ma_vals[x_idx]):
        lines.append(f"MA: {float(ma_vals[x_idx]):.3f}")
    if not lines:
        return
    axs.text(
        0.98,
        0.02,
        "\n".join(lines),
        transform=axs.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        bbox={
            "boxstyle": "round,pad=0.3",
            "facecolor": "white",
            "alpha": 0.85,
            "edgecolor": "black",
        },
    )


def process_relevance_for_plotting(relevance: np.ndarray) -> np.ndarray:
    """Ensure 2D array for plotting."""
    if hasattr(relevance, "dim") and relevance.dim() == 0:
        relevance = relevance.unsqueeze(0)
    if relevance.ndim == 1:
        relevance = relevance.reshape(1, -1)
    return relevance


def minmax_normalize_relevance(relevance: np.ndarray) -> np.ndarray:
    """Minmax transform: positives to [0, 1], negatives to [-1, 0]."""
    relevance = np.array(relevance)
    normalized = np.zeros_like(relevance)

    positive_mask = relevance > 0
    if np.any(positive_mask):
        max_positive = np.max(relevance[positive_mask])
        if max_positive > 0:
            normalized[positive_mask] = relevance[positive_mask] / max_positive

    negative_mask = relevance < 0
    if np.any(negative_mask):
        min_negative = np.min(relevance[negative_mask])
        if min_negative < 0:
            normalized[negative_mask] = relevance[negative_mask] / (-min_negative)

    return normalized


def plot_relevance_heatmap(
    axs,
    relevance: np.ndarray,
    title: str,
    vmin: float = -1.0,
    vmax: float = 1.0,
    cmap: str = "coolwarm",
    colorbar_label: str = "Normalized relevance",
) -> None:
    """Plot heatmap with colorbar and title."""
    image = axs.imshow(
        relevance,
        aspect="auto",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    axs.autoscale(False)
    axs.get_yaxis().set_visible(False)
    cbar = plt.colorbar(image, ax=axs, label=colorbar_label)
    # Increase colorbar tick and label size
    cbar.ax.tick_params(labelsize=14)
    cbar.set_label(colorbar_label, size=14)
    # Larger, bold subplot title
    axs.set_title(title, fontsize=14, fontweight="bold")


def configure_plot_appearance(
    axs, seq: str, explainer_name: str, sequence_length: int
) -> None:
    """Configure axes for single plot."""
    axs.set_title(f"{explainer_name.upper()} Relevance Values")

    axs.set_xlim(0, sequence_length)

    actual_end = len(seq) - 1
    axs.set_xticks([0, actual_end, sequence_length - 1])
    axs.set_xticklabels([0, actual_end, sequence_length - 1])
    axs.tick_params(axis="x", labelsize=14)

    if actual_end < sequence_length - 1:
        axs.axvline(x=actual_end, color="black", linestyle="--", linewidth=2, alpha=0.7)

    legend_elements = [
        Rectangle(
            (0, 0),
            1,
            1,
            facecolor="none",
            edgecolor="black",
            linewidth=1,
            label=f"Sequence length: {len(seq)} positions",
        )
    ]
    axs.legend(handles=legend_elements, loc="upper right", framealpha=0.9)


def configure_plot_appearance_multi(axs, seq: str, sequence_length: int) -> None:
    """Configure axes for multi plot."""
    axs.set_xlim(0, sequence_length)

    actual_end = len(seq) - 1
    axs.set_xticks([0, actual_end, sequence_length - 1])
    axs.set_xticklabels([0, actual_end, sequence_length - 1])
    axs.tick_params(axis="x", labelsize=14)

    if actual_end < sequence_length - 1:
        axs.axvline(x=actual_end, color="black", linestyle="--", linewidth=2, alpha=0.7)


def add_cdr_annotations(
    axs: plt.Axes,
    cdrs: List[List[int]],
    x_idx: int = 0,
    box_height: float = 0.05,
    box_y_position: float = 0.475,
) -> None:
    """
    Ajoute des annotations de CDR à un plot en utilisant des rectangles et du texte.

    Paramètres
    ----------
    axs : plt.Axes
        L'objet axes sur lequel ajouter les annotations.
    cdrs : List[List[int]]
        Une liste de positions de début et de fin de CDR.
    x_idx : int
        L'index de l'image pour laquelle les CDR sont affichés.
    box_height : float
        La hauteur du rectangle d'annotation.
    box_y_position : float
        La position verticale de l'annotation (entre 0 et 1).
    """
    cdr_names = ["CDR1", "CDR2", "CDR3"]
    colors = ["yellow", "orange", "lime"]  # Définissez vos couleurs ici

    if cdrs is not None and x_idx < len(cdrs):
        cdr_positions = cdrs[x_idx]
        if cdr_positions is not None:
            for i, (start, end) in enumerate(
                zip(cdr_positions[::2], cdr_positions[1::2])
            ):
                if i < len(cdr_names):
                    # Calcule la largeur et le centre de la région
                    cdr_width = end - start
                    center_x = (start + end) / 2
                    # Calcule la position y du texte au milieu du rectangle
                    text_y_position = box_y_position + 3 / 2 * box_height

                    # Dessine un rectangle pour le fond de l'annotation
                    # Les coordonnées de départ sont (x, y) du coin inférieur gauche
                    rect = Rectangle(
                        (start, box_y_position),
                        cdr_width,
                        box_height,
                        facecolor=colors[i],
                        edgecolor="black",
                        linewidth=1,
                        alpha=0.6,
                        clip_on=False,
                    )

                    axs.add_patch(rect)

                    # Place le texte au centre du rectangle
                    axs.text(
                        center_x,
                        text_y_position,
                        cdr_names[i],
                        ha="center",
                        va="center",
                        fontweight="bold",
                        fontsize=10,
                        color="black",
                    )
