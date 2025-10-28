"""
Utility functions for Neptune logging operations.
"""

import os
from typing import Any, Optional
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import torch
from neptune import Run as NeptuneRun


def log_metrics_to_neptune(
    metrics: dict[str, Any], neptune_run: Optional[NeptuneRun], dataset_type: str
) -> None:
    """
    Log metrics to Neptune for a specific dataset type.

    Parameters:
    -----------
    metrics (dict[str, Any]):
        Dictionary containing metric names and values to log.
    neptune_run (NeptuneRun | None):
        Neptune run object for logging. If None, does nothing.
    dataset_type (str):
        Type of dataset (e.g., 'train', 'valid', 'test').
    """     
    for metric_name, metric_value in metrics.items():
        # Convert numpy arrays and other non-supported types
        if isinstance(metric_value, np.ndarray):
            # For numpy arrays, convert to list or scalar
            if metric_value.size == 1:
                metric_value = float(metric_value.item())
            else:
                metric_value = metric_value.tolist()
        elif isinstance(metric_value, (np.integer, np.floating)):
            # Convert numpy scalars to Python types
            metric_value = metric_value.item()
        elif isinstance(metric_value, torch.Tensor):
            # Convert PyTorch tensors
            if metric_value.numel() == 1:
                metric_value = float(metric_value.item())
            else:
                metric_value = metric_value.detach().cpu().numpy().tolist()
        if isinstance(neptune_run, dict):
            if f"{dataset_type}/{metric_name}" not in neptune_run:
                neptune_run[f"{dataset_type}/{metric_name}"] = []
            neptune_run[f"{dataset_type}/{metric_name}"].append(metric_value)
        elif isinstance(neptune_run, NeptuneRun):
            neptune_run[f"{dataset_type}/{metric_name}"].log(metric_value)
        else:
            raise ValueError(f"Invalid Neptune run type: {type(neptune_run)}")


def set_author_neptune_api_token() -> None:
    """Set the variable NEPTUNE_API_TOKEN based on the email of commit author.

    It is useful on AIchor to have proper owner of each run.
    """
    try:
        author_email = os.environ["VCS_AUTHOR_EMAIL"]
    # we are not on AIchor
    except KeyError:
        return

    author_email, _ = author_email.split("@")
    author_email = author_email.replace("-", "_").replace(".", "_").upper()

    try:
        author_api_token = os.environ[f"{author_email}__NEPTUNE_API_TOKEN"]
        os.environ["NEPTUNE_API_TOKEN"] = author_api_token
    except KeyError:
        print(f"Neptune credentials for user {author_email} not found.")


def create_graph_run_from_dict(neptune_run: dict) -> None:
    """Create graph for every metric from a dictionary if Neptune is not used.

    Parameters:
    -----------
    neptune_run (dict):
        Dictionary containing the Neptune run.
    """
    #Create a new folder in graphs directory with the name of the run.
    #Name would be the date and time of the run in the format YYYYMMDD_HHMMSS.
    run_name = datetime.utcnow().isoformat().replace(":", "").replace(".", "_")
    os.makedirs(f"graphs/run_{run_name}", exist_ok=True)
    
    # Create a figure for each metric
    for metric_name, metric_values in neptune_run.items():
        #check if metric_values is not full of nan, if so, skip the plot
        if np.all(np.isnan(metric_values)):
            continue
        metric_name = metric_name.replace("/", "_")
        plt.plot(metric_values, label=metric_name)
        plt.legend()
        plt.title(metric_name)
        plt.xlabel("Epoch")
        plt.ylabel(metric_name)
        plt.savefig(f"graphs/run_{run_name}/{metric_name}.png")
        plt.close()

    # Create a figure for the epoch duration
    plt.figure()
    plt.plot(neptune_run["epoch_duration"], label="Epoch duration")
    plt.legend()
    plt.title("Epoch duration")
    plt.xlabel("Epoch")
    plt.ylabel("Epoch duration")
    plt.savefig(f"graphs/run_{run_name}/epoch_duration.png")
    plt.close()