# pylint: disable=too-many-locals, too-many-arguments, too-many-positional-arguments

import matplotlib.pyplot as plt
import numpy as np
from neptune import Run as NeptuneRun
from scipy.stats import spearmanr


def _get_plots_config(data: dict[str, np.ndarray]) -> list[tuple]:
    """
    Get the configuration for all plots to be generated.

    Args:
        data: Dictionary containing plot data

    Returns:
        List of tuples with (function, args, kwargs, condition)
    """
    # Base plot configuration
    base_plots = [
        (
            _plot_binding_probability_rank_subplot,
            [data["binding_probs_SN"]],
            {},
            "binding_probs_SN" in data,
        ),
        (
            _plot_predictions_with_error_bars_subplot,
            [
                data["predictions_SN"],
                data["counts_S"],
                data["binding_probs_SN"],
                data["initial_counts_S"],
            ],
            {},
            "predictions_SN" in data
            and "counts_S" in data
            and "binding_probs_SN" in data
            and "initial_counts_S" in data,
        ),
        (
            _plot_binding_selectivity_subplot,
            [data["binding_probs_SN"], data["selectivities_S"]],
            {
                "counts": {
                    "initial_counts_S": data["initial_counts_S"],
                    "selected_counts_S": data["counts_S"],
                },
                "x_label": "Normalized Selectivity",
                "y_label": "Binding Probability",
            },
            "binding_probs_SN" in data
            and "selectivities_S" in data
            and "initial_counts_S" in data
            and "counts_S" in data,
        ),
        (
            _plot_binding_selectivity_subplot,
            [data["binding_probs_SN"], data["selectivities_S"]],
            {
                "counts": {
                    "initial_counts_S": data["initial_counts_S"],
                    "selected_counts_S": data["counts_S"],
                },
                "without_0_selectivity": True,
                "x_label": "Normalized Selectivity without 0 selectivity",
                "y_label": "Binding Probability",
            },
            "binding_probs_SN" in data
            and "selectivities_S" in data
            and "initial_counts_S" in data
            and "counts_S" in data,
        ),
    ]

    # Check if negative binding probabilities exist
    has_negative_binding_probs = "negative_binding_probs_S" in data and not np.all(
        data["negative_binding_probs_S"] == 0
    )

    if has_negative_binding_probs:
        # Additional plots for negative binding probabilities
        neg_binding_plots = [
            (
                _plot_binding_selectivity_subplot,
                [data["negative_binding_probs_S"], data["selectivities_S"]],
                {
                    "counts": {
                        "initial_counts_S": data["initial_counts_S"],
                        "selected_counts_S": data["counts_S"],
                    },
                    "x_label": "Negative Binding Probability",
                    "y_label": "Selectivity",
                },
                "selectivities_S" in data
                and "initial_counts_S" in data
                and "counts_S" in data,
            ),
            (
                _plot_binding_selectivity_subplot,
                [
                    data["binding_probs_SN"] * (1 - data["negative_binding_probs_S"]),
                    data["selectivities_S"],
                ],
                {
                    "counts": {
                        "initial_counts_S": data["initial_counts_S"],
                        "selected_counts_S": data["counts_S"],
                    },
                    "x_label": "Binding Prob * (1 - Negative Binding Prob)",
                    "y_label": "Selectivity",
                },
                "selectivities_S" in data
                and "initial_counts_S" in data
                and "counts_S" in data,
            ),
            (
                _plot_binding_selectivity_subplot,
                [data["negative_binding_probs_S"], data["selectivities_S"]],
                {
                    "counts": {
                        "initial_counts_S": data["initial_counts_S"],
                        "selected_counts_S": data["counts_S"],
                    },
                    "without_0_selectivity": True,
                    "x_label": "Negative Binding Probability without 0 selectivity",
                    "y_label": "Selectivity",
                },
                "selectivities_S" in data
                and "initial_counts_S" in data
                and "counts_S" in data,
            ),
            (
                _plot_binding_selectivity_subplot,
                [
                    data["binding_probs_SN"] * (1 - data["negative_binding_probs_S"]),
                    data["selectivities_S"],
                ],
                {
                    "counts": {
                        "initial_counts_S": data["initial_counts_S"],
                        "selected_counts_S": data["counts_S"],
                    },
                    "without_0_selectivity": True,
                    "x_label": "Binding Prob * (1 - Negative Binding Prob) "
                    "without 0 selectivity",
                    "y_label": "Selectivity",
                },
                "selectivities_S" in data
                and "initial_counts_S" in data
                and "counts_S" in data,
            ),
        ]
        base_plots.extend(neg_binding_plots)

    return base_plots


def scatter_plot_predictions(
    data: dict[str, np.ndarray],
    neptune_run: NeptuneRun,
    dataset_type: str,
    name: str,
    epoch: int,
) -> None:
    """
    Create a comprehensive scatter plot with subplots.

    Parameters:
    -----------
    data: dict[str, np.ndarray]
        Dictionary containing data to plot with keys:
        - 'predictions': predictions array (n_samples, n_sequences)
        - 'counts': truth counts array (n_sequences,)
        - 'binding_probs': binding probabilities array (n_samples, n_sequences)
        - 'selectivities': selectivities array (n_sequences,)
        - 'initial_counts': initial counts array (n_sequences,)
        - 'negative_binding_probs': negative binding probabilities array
        (n_samples, n_sequences)
    neptune_run: NeptuneRun
        Neptune run object.
    dataset_type: str
        Dataset type (e.g., "train", "val", "test").
    name: str
        Experiment/model name.
    epoch: int
        Current training epoch.
    """
    # Determine if negative binding probabilities exist
    has_negative_binding_probs = "negative_binding_probs_S" in data and not np.all(
        data["negative_binding_probs_S"] == 0
    )

    # Create figure with dynamic layout
    fig, axes = plt.subplots(
        *(2, 4) if has_negative_binding_probs else (1, 4),
        figsize=(18, 12) if has_negative_binding_probs else (18, 6),
    )
    fig.suptitle(
        f"{name} - Comprehensive Analysis (Epoch {epoch}, {dataset_type} set)",
        fontsize=16,
    )

    # Generate plots
    for i, (plot_func, args, kwargs, condition) in enumerate(_get_plots_config(data)):
        if i < len(axes.flatten()) and condition:
            plot_func(*args, ax=axes.flatten()[i], **kwargs)

    plt.tight_layout()

    # Upload to Neptune
    neptune_run[
        f"plots/{dataset_type}_{name}_comprehensive_scatter_epoch_{epoch}"
    ].upload(
        fig, description=f"Comprehensive analysis for {name} on {dataset_type} set"
    )

    plt.close(fig)


def _plot_predictions_with_error_bars_subplot(
    predictions: np.ndarray,
    truth_counts: np.ndarray,
    binding_probs: np.ndarray = None,
    initial_counts: np.ndarray = None,
    ax: plt.Axes = None,
) -> None:
    """
    Create predictions vs truth subplot with error bars in log scale.
    """
    # Calculate prediction statistics
    predictions_squeezed = predictions.squeeze()
    pred_stats = {
        "mean": np.mean(predictions_squeezed, axis=1).flatten(),
        "std": np.std(predictions_squeezed, axis=1).flatten(),
    }
    # Calculate correlation
    spearman_corr = spearmanr(pred_stats["mean"], truth_counts)[0]
    # Prepare visualization data
    viz_data = {
        "colors": np.log10(np.mean(binding_probs, axis=0)),
        "sizes": np.clip(100 * np.log10(initial_counts), 10, 100),
        "count_errors": np.sqrt(initial_counts),
    }

    # Create scatter plot
    scatter = ax.scatter(
        truth_counts,
        pred_stats["mean"],
        c=viz_data["colors"],
        cmap="viridis",
        alpha=0.6,
        s=viz_data["sizes"],
        label="Average preds number",
        zorder=2,
    )

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
    cbar.set_label("log10(Binding Probability)", rotation=270, labelpad=15)

    # Add legend for point sizes
    if initial_counts is not None:
        size_range = {
            "min": np.clip(100 * np.log10(initial_counts.min()), 10, 100),
            "max": np.clip(100 * np.log10(initial_counts.max()), 10, 100),
        }

        ax.scatter(
            [],
            [],
            s=size_range["min"],
            c="gray",
            alpha=0.6,
            label=f"Small initial count\n({initial_counts.min():.0f})",
        )
        ax.scatter(
            [],
            [],
            s=size_range["max"],
            c="gray",
            alpha=0.6,
            label=f"Large initial count\n({initial_counts.max():.0f})",
        )

    # Add error bars
    ax.errorbar(
        truth_counts,
        pred_stats["mean"],
        xerr=viz_data["count_errors"],
        yerr=pred_stats["std"],
        fmt="none",
        alpha=0.6,
        capsize=3,
        capthick=1,
        ecolor="black",
        elinewidth=0.5,
        zorder=1,
    )

    # Add diagonal line for perfect correlation
    plot_range = {
        "min": min(truth_counts.min(), pred_stats["mean"].min()),
        "max": max(truth_counts.max(), pred_stats["mean"].max()),
    }
    ax.plot(
        [plot_range["min"], plot_range["max"]],
        [plot_range["min"], plot_range["max"]],
        "r--",
        alpha=0.8,
        label="Perfect correlation",
    )

    ax.set_xlabel("Truth number count after selection (counting error)")
    ax.set_ylabel("Average preds number after selection (std)")
    ax.set_title(f"Predictions vs Truth\nSpearman r = {spearman_corr:.3f}")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    # Set log scale for both axes
    ax.loglog()


def _plot_binding_probability_rank_subplot(
    binding_probs: np.ndarray,
    ax: plt.Axes,
) -> None:
    """
    Create binding probability vs rank subplot in log scale.
    """
    # Calculate mean and std of binding probabilities
    mean_binding = np.mean(binding_probs, axis=0).squeeze()
    std_binding = np.std(binding_probs, axis=0).squeeze()

    # Sort binding probabilities in descending order
    sorted_indices = np.argsort(mean_binding)[::-1]
    sorted_binding = mean_binding[sorted_indices]
    sorted_std = std_binding[sorted_indices]
    ranks = np.arange(1, len(sorted_binding) + 1)

    # Add error bars first
    ax.errorbar(
        ranks,
        sorted_binding,
        yerr=sorted_std,
        fmt="none",  # No points here, just error bars
        alpha=0.8,
        capsize=2,
        capthick=0.5,
        elinewidth=0.5,
        ecolor="black",
    )

    # Then add the points on top
    ax.plot(
        ranks,
        sorted_binding,
        "ro",  # Red dots
        markersize=4,
        label="Binding probability (std)",
    )

    ax.loglog()

    ax.set_xlabel("Rank")
    ax.set_ylabel("Binding Probability")
    ax.set_title("Binding Probability vs Rank")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)


def calculate_selectivity_errors(
    selectivities: np.ndarray,
    initial_counts: np.ndarray,
    selected_counts: np.ndarray,
    epsilon: float = 1e-10,
) -> tuple[np.ndarray, np.ndarray,]:  # normalized selectivities  # selectivity errors
    """Calculate normalized selectivities and their associated errors.

    Args:
        selectivities (np.ndarray): Raw selectivity values
        initial_counts (np.ndarray): Initial count values
        selected_counts (np.ndarray): Selected count values
        epsilon (float, optional): Small value to avoid division by zero.
        Defaults to 1e-10.

    Returns:
        tuple containing:
            - np.ndarray: Normalized selectivities
            - np.ndarray: Associated error values
    """
    # Convert inputs to numpy arrays
    selectivities = np.asarray(selectivities).squeeze()
    initial_counts = np.asarray(initial_counts)
    selected_counts = np.asarray(selected_counts)

    # Handle zero values in selectivities
    selectivities = np.where(selectivities > 0, selectivities, epsilon)

    # Normalize selectivities

    normalized_select = selectivities / selectivities.sum()

    # Calculate errors with safe values
    safe_initial_counts = np.where(initial_counts > 0, initial_counts, epsilon)
    safe_selected_counts = np.where(selected_counts > 0, selected_counts, epsilon)

    sqrt_initial = 1 / np.sqrt(safe_initial_counts)
    sqrt_selected = 1 / np.sqrt(safe_selected_counts)
    error_calculation = sqrt_initial + sqrt_selected

    selectivity_errors = np.where(
        (initial_counts > 0) & (selected_counts > 0),
        error_calculation * normalized_select,
        0.0,  # Set error to 0 for invalid points
    )

    return normalized_select, selectivity_errors


def _plot_binding_selectivity_subplot(
    binding_probs: np.ndarray,
    selectivities: np.ndarray,
    counts: dict[str, np.ndarray] = None,
    without_0_selectivity: bool = False,
    x_label: str = "Normalized Selectivity",
    y_label: str = "Binding Probability",
    ax: plt.Axes = None,
) -> None:
    """
    Create binding probability vs selectivity subplot in log scale.
    """
    if counts is None:
        counts = {"initial_counts_S": None, "selected_counts_S": None}

    # Calculate mean and std of binding probabilities
    mean_binding = np.mean(binding_probs, axis=0).squeeze()
    std_binding = np.std(binding_probs, axis=0).squeeze()

    number_of_points = len(mean_binding)
    number_of_points_without_0_selectivity = len(mean_binding[selectivities > 0])

    if without_0_selectivity:
        counts["initial_counts_S"] = counts["initial_counts_S"][selectivities > 0]
        counts["selected_counts_S"] = counts["selected_counts_S"][selectivities > 0]
        mean_binding = mean_binding[selectivities > 0]
        std_binding = std_binding[selectivities > 0]
        selectivities = selectivities[selectivities > 0]

    # Calculate normalized selectivities and their errors
    normalized_select, selectivity_errors = calculate_selectivity_errors(
        selectivities=selectivities,
        initial_counts=counts["initial_counts_S"],
        selected_counts=counts["selected_counts_S"],
    )

    # Calculate Spearman correlation
    spearman_corr = spearmanr(mean_binding, selectivities)[0]

    # Create scatter plot with error bars - red points and thin black error bars
    ax.errorbar(
        normalized_select,
        mean_binding,
        xerr=selectivity_errors,
        yerr=std_binding,
        fmt="ro",  # Red dots
        alpha=0.8,
        capsize=2,  # Thinner caps
        capthick=0.5,  # Thinner cap thickness
        elinewidth=0.5,  # Thinner error bar lines
        ecolor="black",  # Black error bars
        markersize=4,  # Slightly smaller markers
        label=(
            "Binding probability (std)"
            if without_0_selectivity
            else (
                f"Binding probability (std) "
                f"({number_of_points_without_0_selectivity}/{number_of_points} "
                f"of 0 dropped)"
            )
        ),
    )

    # Add diagonal line (x = y)
    min_val = min(normalized_select.min(), mean_binding.min())
    max_val = max(normalized_select.max(), mean_binding.max())
    ax.plot(
        [min_val, max_val],
        [min_val, max_val],
        "b--",
        alpha=0.8,
        label="x = y",
    )

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(
        f"Binding Probability vs Selectivity\nSpearman r = {spearman_corr:.3f}"
    )
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    # Set log scale for both axes
    ax.loglog()
