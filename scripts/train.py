# pylint: disable=E1120
import os
from datetime import datetime
from pathlib import Path

import hydra
import neptune

from src.data_processing.prepare_dataset import prepare_dataset_for_training
from src.models.training_helpers import train_bnn
from src.utils.calculate_null_model import calculate_null_model_correlation
from src.utils.device import select_device
from src.utils.extract_experience_name import extract_experience_name
from src.utils.local_io import get_data_dir, get_results_dir, save_yaml_local
from src.utils.neptune_utils import set_author_neptune_api_token
from src.utils.seed import set_seed

script_dir = os.path.dirname(os.path.abspath(__file__))
os.environ["TOKENIZERS_PARALLELISM"] = "false"

config_path = os.path.join(script_dir, "../config")
absolute_config_path = os.path.abspath(config_path)


def setup_locations(cfg, run_id="none"):
    """
    Setup all file and directory locations needed for training.

    Parameters
    ----------
    cfg : DictConfig
        Configuration object containing paths and experiment settings

    Returns
    -------
    dict
        Dictionary containing all relevant paths
    """
    data_base: Path = get_data_dir(cfg)
    results_base: Path = get_results_dir(cfg)

    if cfg.model.training.relative_csv_file_location is None:
        csv_file_location = extract_experience_name(
            cfg.model.splits.training_pair[0],
            cfg.model.splits.relative_csv_file_location,
        )
    else:
        csv_file_location = cfg.model.training.relative_csv_file_location

    locations = {
        "data_base": data_base,
        "results_base": results_base,
        "csv_file": data_base / csv_file_location,
        "experiments_dir": results_base / "experiments",
    }

    # Setup experiment directory
    locations["experiment_dir"] = locations["experiments_dir"] / "_".join(
        [
            cfg.model.training.experiment_name,
            datetime.utcnow().isoformat().replace(":", ""),
        ]
        + [run_id]
    )
    locations["experiment_dir"].mkdir(parents=True, exist_ok=True)

    # Setup batches directory
    locations["batches_dir"] = locations["experiment_dir"] / "batches"
    locations["batches_dir"].mkdir(parents=True, exist_ok=True)

    # Setup model checkpoint directory
    relative_model_checkpoint_location = (
        cfg.model.training.relative_model_checkpoint_location
    )
    if relative_model_checkpoint_location is not None:
        locations["model_checkpoint_dir"] = (
            locations["experiments_dir"] / relative_model_checkpoint_location
        )
    else:
        locations["model_checkpoint_dir"] = None

    return locations


@hydra.main(version_base=None, config_path=absolute_config_path, config_name="default")
def main(cfg):
    """
    Main function to train either the baseline or the FLIGHTED model.

    Parameters:
    -----------
    cfg (DictConfig):
        Configuration object.
    """
    set_author_neptune_api_token()
    set_seed()
    device = select_device(cfg)

    # Initialize Neptune and calculate null model correlation
    neptune_run = None
    run_id = "none"
    if cfg.use_neptune:
        neptune_run = neptune.init_run(
            project=cfg.neptune_project, tags=list(cfg.model.tags)
        )
        run_id = neptune_run["sys/id"].fetch()
        print(f"Neptune run ID: {run_id}")
    else:
        neptune_run = {}

    # Setup all locations
    locations = setup_locations(cfg, run_id)

    # Prepare dataset
    (
        _,
        _,
        train_dataloader,
        valid_dataloader,
        test_dataloader,
        kd_values,
        embedding_size,
    ) = prepare_dataset_for_training(
        csv_path=locations["csv_file"],
        cfg=cfg,
        batches_path=locations["batches_dir"],
        device=device,
        test_data_path=None,
    )

    # keep the cfg in the experiment directory
    save_yaml_local(cfg, locations["experiment_dir"] / "cfg.yaml")

    null_model_correlation = {}
    for experiment in cfg.model.splits.training_pair:
        null_model_correlation[
            "train/" + experiment
        ] = calculate_null_model_correlation(
            train_dataloader,
            experiment,
            cfg,
        )
    for experiment in cfg.model.splits.validation_pair:
        null_model_correlation[
            "valid/" + experiment
        ] = calculate_null_model_correlation(
            valid_dataloader,
            experiment,
            cfg,
        )
    if cfg.use_neptune and neptune_run is not None:
        neptune_add_info(neptune_run, cfg, null_model_correlation)

    # Train the model
    train_bnn(
        train_dataloader,
        valid_dataloader,
        test_dataloader,
        cfg,
        locations["experiment_dir"],
        locations["model_checkpoint_dir"],
        neptune_run,
        device=device,
        embedding_size=embedding_size,
        kd_values=kd_values,
    )
    if cfg.use_neptune and neptune_run is not None:
        neptune_run.stop()


def neptune_add_info(neptune_run, cfg, null_model_correlation):
    """
    Add information to the neptune run.
    """
    neptune_run["parameters"] = dict(cfg)
    neptune_run["data_train"] = "/".join(cfg.model.splits.training_pair)
    neptune_run["data_valid"] = "/".join(cfg.model.splits.validation_pair)
    for key, value in null_model_correlation.items():
        # Create a constant curve by adding the same value for each epoch
        for _ in range(cfg.model.training.epochs):
            neptune_run[f"{key}/null_model_correlation"].append(value)
    if cfg.model.fitness_predictor_architecture == "CNN":
        neptune_run["layers"] = (
            "CNN "
            + " ".join(
                str(channel)
                for channel in cfg.model.fitness_predictor_cnn.list_of_channels
            )
            + " Kernel "
            + " ".join(
                str(kernel)
                for kernel in cfg.model.fitness_predictor_cnn.list_of_kernel_size
            )
            + " Stride "
            + " ".join(
                str(stride) for stride in cfg.model.fitness_predictor_cnn.list_of_stride
            )
            + " Padding"
            + " ".join(
                str(padding)
                for padding in cfg.model.fitness_predictor_cnn.list_of_padding
            )
            + " MLP "
            + " ".join(
                str(layer) for layer in cfg.model.fitness_predictor_cnn.layers_size
            )
        )
    elif cfg.model.fitness_predictor_architecture == "MLP":
        neptune_run["layers"] = "MLP " + " ".join(
            str(layer) for layer in cfg.model.fitness_predictor_mlp.layers_size
        )


if __name__ == "__main__":
    main()
