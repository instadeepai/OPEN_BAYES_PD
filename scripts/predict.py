# pylint: disable=E1120,R0915,R0914
import os
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

import hydra
import neptune
import numpy as np
import pandas as pd
from neptune.types import File
from omegaconf import OmegaConf

from src.data_processing.dataloader import PhageDisplayDataLoader
from src.data_processing.dataset import SelectionDataset
from src.data_processing.prepare_dataset import (
    adding_columns_to_df_for_pd_prediction,
    correct_counts,
    handle_kd_dataframe,
)
from src.models.predict_helpers import predict_bnn
from src.utils.device import select_device
from src.utils.local_io import (
    get_data_dir,
    get_results_dir,
    load_csv_local,
    load_yaml_local,
    save_npy_local,
)
from src.utils.neptune_utils import set_author_neptune_api_token
from src.utils.seed import set_seed

script_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(script_dir, "../config")
absolute_config_path = os.path.abspath(config_path)


def _initialize_environment(cfg: Any) -> Tuple[Any, Any]:
    """
    Initialize the environment for prediction.

    Args:
        cfg: Configuration object.

    Returns:
        Tuple containing device and data_base.
    """
    set_author_neptune_api_token()
    device = select_device(cfg)
    set_seed()
    data_base = get_data_dir(cfg)

    return device, data_base


def _create_experiment_directory(results_base: Any, cfg: Any) -> Any:
    """
    Create experiment directory with timestamp.

    Args:
        results_base: Base results path.
        cfg: Configuration object.

    Returns:
        Experiment directory path.
    """
    path_to_experiments_dir = results_base / "experiments"

    # Build experiment name components
    name_parts = [
        cfg.model.prediction.experiment_name,
        datetime.utcnow().isoformat().replace(":", ""),
    ]

    # Add tags if they exist and are iterable
    if hasattr(cfg.model, "tags") and cfg.model.tags:
        try:
            # Check if tags is iterable and not a string
            is_iterable = hasattr(cfg.model.tags, "__iter__")
            is_not_string = not isinstance(cfg.model.tags, str)
            if is_iterable and is_not_string:
                name_parts.append("-".join(cfg.model.tags))
            else:
                name_parts.append(str(cfg.model.tags))
        except (TypeError, AttributeError) as error:
            print(f"Error adding tags: {error}")

    experiment_dir = path_to_experiments_dir / "_".join(name_parts)
    experiment_dir.mkdir(parents=True, exist_ok=True)
    return experiment_dir


def _load_and_prepare_predict_data(
    data_base: Any, cfg: Any, device: Any
) -> Tuple[pd.DataFrame, SelectionDataset, PhageDisplayDataLoader, int]:
    """
    Load and prepare prediction data.

    Args:
        data_base: Base data path.
        cfg: Configuration object.
        device: Device for computation.

    Returns:
        Tuple containing predict_df, predict_dataset, predict_dataloader,
        and embedding_size.
    """
    path_to_csv_file_predict = (
        data_base / cfg.model.prediction.to_predict_csv_file_location
    )
    predict_df = load_csv_local(path_to_csv_file_predict)
    predict_df = adding_columns_to_df_for_pd_prediction(predict_df)

    if "VHH Sequence (AA)" in predict_df.columns:
        predict_df = handle_kd_dataframe(predict_df)

    # Determine embedding size
    if hasattr(predict_df, "embedding_size"):
        embedding_size = predict_df.embedding_size()
    else:
        embedding_size = cfg.model.embedding_size

    predict_dataset = SelectionDataset(
        correct_counts(predict_df), "null", "null", cfg, device, True
    )
    predict_dataloader = PhageDisplayDataLoader(
        predict_dataset,
        cfg.model.prediction.experiment_pair_predict,
        batch_size=10000,
        shuffle=False,
    )

    return predict_df, predict_dataset, predict_dataloader, embedding_size


def _load_background_data(
    data_base: Any, cfg: Any, device: Any
) -> Optional[PhageDisplayDataLoader]:
    """
    Load background data if specified in configuration.

    Args:
        data_base: Base data path.
        cfg: Configuration object.
        device: Device for computation.

    Returns:
        Background dataloader or None if not specified.
    """
    if cfg.model.prediction.background_csv_file_location is None:
        return None

    path_to_csv_file_background = (
        data_base / cfg.model.prediction.background_csv_file_location
    )
    background_df = load_csv_local(path_to_csv_file_background)
    background_df = adding_columns_to_df_for_pd_prediction(background_df)
    background_dataset = SelectionDataset(
        correct_counts(background_df), "null", "null", cfg, device, True
    )
    background_dataloader = PhageDisplayDataLoader(
        background_dataset, cfg.model.prediction.experiment_pair_background
    )

    return background_dataloader


def _save_explanation_results(
    output_dict: Dict[str, Any], experiment_dir: Any, cfg: Any
) -> None:
    """
    Save explanation results (SHAP and CAM) to Neptune

    Args:
        output_dict: Dictionary containing prediction results.
        experiment_dir: Experiment directory path.
        cfg: Configuration object.
    """
    print("output_dict", experiment_dir)
    if cfg.use_neptune and cfg.model.explainer.use_neptune:
        neptune_run = neptune.init_run(project=cfg.neptune_project)

    explainer_name = cfg.model.explainer.type

    for image in os.listdir(output_dict[f"address_{explainer_name}_save_images"]):
        image_path = output_dict[f"address_{explainer_name}_save_images"] + f"/{image}"
        os.makedirs(
            experiment_dir / "explainer_images" / f"explainer_{explainer_name}_images",
            exist_ok=True,
        )
        with (
            open(image_path, "rb") as fsrc,
            open(
                experiment_dir
                / "explainer_images"
                / f"explainer_{explainer_name}_images/{image}",
                "wb",
            ) as fdst,
        ):
            fdst.write(fsrc.read())

        if cfg.use_neptune and cfg.model.explainer.use_neptune:
            neptune_run["prediction_images"].append(File(image_path))


def _save_prediction_results(
    output_dict: Dict[str, Any], predict_df: pd.DataFrame, experiment_dir: Any
) -> None:
    """
    Save prediction results to files.

    Args:
        output_dict: Dictionary containing prediction results.
        predict_df: Prediction dataframe.
        experiment_dir: Experiment directory path.
    """
    print("experiment_dir", experiment_dir)
    # Save raw prediction arrays
    save_npy_local(
        np.array(output_dict["preds_proba_mean"]),
        experiment_dir / "mean_prediction.npy",
    )
    save_npy_local(
        np.array(output_dict["preds_proba_std"]), experiment_dir / "std_prediction.npy"
    )
    save_npy_local(np.array(output_dict["sequence"]), experiment_dir / "sequence.npy")

    # only the experiments in experiment_pair_predict

    # Create and save predictions DataFrame
    predictions_df = pd.DataFrame(
        {
            "sequence": np.array(output_dict["sequence"]),
            "prediction_mean": np.array(output_dict["preds_proba_mean"]).squeeze(),
            "prediction_std": np.array(output_dict["preds_proba_std"]).squeeze(),
            "prediction_neg_mean": np.array(
                output_dict["preds_proba_neg_mean"]
            ).squeeze(),
            "prediction_neg_std": np.array(
                output_dict["preds_proba_neg_std"]
            ).squeeze(),
            "prediction_target_mean": np.array(
                output_dict["preds_proba_target_mean"]
            ).squeeze(),
            "prediction_target_std": np.array(
                output_dict["preds_proba_target_std"]
            ).squeeze(),
        }
    )
    if "auc_ma_metrics" in output_dict:
        if "union" in output_dict["auc_ma_metrics"]:
            metrics_to_extract = {
                "mean": ["auc_roc", "mass_accuracy"],
                "union": ["auc_roc", "mass_accuracy"],
                "intersection": ["auc_roc", "mass_accuracy"],
                "uai_plus": ["auc_roc", "mass_accuracy"],
            }
        else:
            metrics_to_extract = {
                "mean": ["auc_roc", "mass_accuracy"],
            }
        for metric_type, names in metrics_to_extract.items():
            if names is None:
                continue
            for metric_name in names:
                column_name = (
                    f"{'auc' if metric_name == 'auc_roc' else 'ma'}_{metric_type}"
                )
                metric_value = output_dict["auc_ma_metrics"][metric_type][metric_name]
                predictions_df[column_name] = metric_value

    predictions_raw_df = pd.DataFrame(
        np.array(output_dict["preds_proba"]).squeeze(), columns=output_dict["sequence"]
    )

    csv_output_path = experiment_dir.joinpath("predictions.csv")
    predictions_df.to_csv(csv_output_path, index=False)

    predictions_raw_df.to_csv(experiment_dir / "predictions_raw.csv", index=False)

    print(f"Prediction results saved to: {csv_output_path}")


@hydra.main(version_base=None, config_path=absolute_config_path, config_name="default")
def main(cfg: Any) -> None:
    """
    Main function to predict using either the baseline or the FLIGHTED model.
    Possibility to explain the results if the baseline model is used with a CNN.

    Args:
        cfg: Configuration object.
    """
    # Initialize environment
    device, data_base = _initialize_environment(cfg)

    # Create experiment directory
    results_base = get_results_dir(cfg)
    experiment_dir = _create_experiment_directory(results_base, cfg)

    if cfg.model.prediction.use_training_cfg:
        cfg = export_cfg_training_args(cfg, results_base)

    # if a cfg.yaml file is in the experiment directory, load it and override the train
    #  parameters so you do not have issues with the model architecture
    print("experiment_dir", experiment_dir)
    if (experiment_dir / "cfg.yaml").exists():
        print("Loading cfg from experiment directory")
        with open(experiment_dir / "cfg.yaml", "r", encoding="utf-8") as f:
            cfg_experiment = OmegaConf.create(f.read())
            cfg.model.training = cfg_experiment.model.training
            cfg.model.fitness_predictor_architecture = (
                cfg_experiment.model.fitness_predictor_architecture
            )
            cfg.model.fitness_predictor_mlp = cfg_experiment.model.fitness_predictor_mlp
            cfg.model.fitness_predictor_cnn = cfg_experiment.model.fitness_predictor_cnn
            cfg.model.splits = cfg_experiment.model.splits

    # Load and prepare prediction data
    (
        predict_df,
        _predict_dataset,
        predict_dataloader,
        embedding_size,
    ) = _load_and_prepare_predict_data(data_base, cfg, device)

    # Load background data if specified
    background_dataloader = _load_background_data(data_base, cfg, device)

    # Perform prediction
    output_dict = predict_bnn(
        predict_dataloader,
        cfg,
        device,
        embedding_size,
        background_dataloader,
    )
    print("done predicting")

    # Save explanation results if enabled
    if cfg.model.prediction.explain:
        _save_explanation_results(output_dict, experiment_dir, cfg)
        print("done explaining")

    # Save prediction results
    _save_prediction_results(output_dict, predict_df, experiment_dir)


def export_cfg_training_args(cfg, data_base):
    """
    Export training arguments from a previous experiment.

    Args:
        cfg: Configuration object.
        data_base: Base data path.

    Returns:
        Configuration object with training arguments overridden.
    """
    # only if there are something in the model_dir_path
    cfg_experiment = load_yaml_local(
        data_base / cfg.model.prediction.model_dir_path / "cfg.yaml"
    )
    print(cfg_experiment.keys())
    # override the cfg with the cfg_experiment
    cfg.model.training = cfg_experiment["model"]["training"]
    cfg.model.fitness_predictor_architecture = cfg_experiment["model"][
        "fitness_predictor_architecture"
    ]
    cfg.model.fitness_predictor_mlp = cfg_experiment["model"]["fitness_predictor_mlp"]
    cfg.model.fitness_predictor_cnn = cfg_experiment["model"]["fitness_predictor_cnn"]

    return cfg


if __name__ == "__main__":
    main()
