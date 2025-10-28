from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig

from src.data_processing.data_split import split_data
from src.data_processing.dataloader import (
    LazyPhageDisplayDataLoader,
    PhageDisplayDataLoader,
)
from src.data_processing.dataset import SelectionDataset
from src.utils.custom_exceptions import NonCompliantDataframeError
from src.utils.enum_definitions import DatasetType
from src.utils.local_io import load_csv_local


def collate_and_move_to_cuda(
    batch: dict[str, torch.Tensor | list[str] | int],
) -> dict[str, torch.Tensor | list[str] | int]:
    """Move tensors in batch to CUDA device."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            batch[key] = value.to(device, non_blocking=True)

    return batch


def prepare_dataset_for_training(
    csv_path: Path,
    cfg: DictConfig,
    device: torch.device,
    batches_path: Path,
    test_data_path: Path,
) -> tuple[
    SelectionDataset,
    SelectionDataset,
    PhageDisplayDataLoader,
    PhageDisplayDataLoader,
    PhageDisplayDataLoader,
    Optional[list[float]],
    tuple[int, int],
]:
    """Prepare the dataset for training.

    Parameters
    ----------
    csv_path: Path
        Path to the CSV file containing the training data.
    cfg: DictConfig
        Configuration object.
    device: torch.device
        Device to use for the dataset.
    batches_path: Path
        Path to the batches directory.
    test_data_path: Path
        Path to the CSV file containing the test data.

    Returns
    -------
    train_dataset: SelectionDataset
        Training dataset.
    valid_dataset: SelectionDataset
        Validation dataset.
    train_dataloader: PhageDisplayDataLoader
        Training dataloader.
    valid_dataloader: PhageDisplayDataLoader
        Validation dataloader.
    test_dataloader: PhageDisplayDataLoader
        Test dataloader.
    kd_values: Optional[list[float]]
        KD values.
    embedding_size: tuple[int, int]
        Embedding size.
    """
    # 1. Core Data Loading, Cleaning, and Splitting
    (
        dataframe_train,
        dataframe_valid,
        dataframe_test,
        kd_values,
    ) = _load_clean_and_split_data(csv_path, cfg, test_data_path)

    # 2. Dataset and Dataloader Instantiation (Centralized)
    (
        train_dataset,
        valid_dataset,
        _,
        dataloader_train,
        dataloader_valid,
        dataloader_test,
    ) = _create_datasets_and_loaders(
        cfg, device, batches_path, (dataframe_train, dataframe_valid, dataframe_test)
    )

    # 3. Final Return
    return (
        train_dataset,
        valid_dataset,
        dataloader_train,
        dataloader_valid,
        dataloader_test,
        kd_values,
        cfg.model.embedding_size,
    )


def _load_clean_and_split_data(
    csv_path: Path, cfg: DictConfig, test_data_path: Optional[Path]
) -> tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame], Optional[pd.Series]]:
    """Loads, cleans, and splits the main training/validation data and loads test data.

    Parameters
    ----------
    csv_path: Path
        Path to the CSV file containing the training data.
    cfg: DictConfig
        Configuration object.
    test_data_path: Path
        Path to the CSV file containing the test data.

    Returns
    -------
    dataframe_train: pd.DataFrame
        Training dataframe.
    dataframe_valid: pd.DataFrame
        Validation dataframe.
    dataframe_test: Optional[pd.DataFrame]
        Test dataframe.
    kd_values: Optional[pd.Series]
        KD values.
    """
    # 1. Load and clean main data
    selection_dataframe = load_csv_local(csv_path)
    if cfg.model.training.delete_initial_null_counts:
        selection_dataframe = handle_initial_null_counts(
            selection_dataframe, cfg.model.training.delete_initial_null_counts
        )
        selection_dataframe = correct_counts(selection_dataframe)

    # 2. Split main data
    dataframe_train, dataframe_valid = split_data(
        dataframe=selection_dataframe,
        training_pairs=cfg.model.splits.training_pair,
        validation_pairs=cfg.model.splits.validation_pair,
    )

    # 3. Handle optional test data
    dataframe_test = None
    kd_values = None
    if test_data_path is not None:
        dataframe_test = load_csv_local(test_data_path)
        kd_values = {
            "KD_values": dataframe_test["KD (M)"],
            "sequence": dataframe_test["VHH Sequence (AA)"],
        }

        # NOTE: The prediction preprocessing logic should ideally be isolated,
        # but is kept here for function completeness.
        if "VHH Sequence (AA)" in dataframe_test.columns:
            dataframe_test = adding_columns_to_df_for_pd_prediction(dataframe_test)
            dataframe_test = handle_kd_dataframe(dataframe_test)
            dataframe_test = correct_counts(dataframe_test)

    return dataframe_train, dataframe_valid, dataframe_test, kd_values


def _create_datasets_and_loaders(
    cfg: DictConfig,
    device: torch.device,
    batches_path: Path,
    dataframes: tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]],
) -> tuple:
    """Instantiates SelectionDatasets and PhageDisplayDataLoaders.

    Parameters
    ----------
    cfg: DictConfig
        Configuration object.
    device: torch.device
        Device to use for the dataset.
    batches_path: Path
        Path to the batches directory.
    dataframes: tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]
        Tuple containing the training dataframe, validation dataframe, and
        optional test dataframe.

    Returns
    -------
    train_dataset: SelectionDataset
        Training dataset.
    valid_dataset: SelectionDataset
        Validation dataset.
    test_dataset: Optional[SelectionDataset]
        Test dataset.
    dataloader_train: PhageDisplayDataLoader
        Training dataloader.
    dataloader_valid: PhageDisplayDataLoader
        Validation dataloader.
    dataloader_test: Optional[PhageDisplayDataLoader]
        Test dataloader.
    """
    # Determine if ESM embeddings need to be generated (if batches_path is empty)
    generate_esm_embeddings = True

    # Create Datasets
    train_dataset = SelectionDataset(
        dataframes[0],
        DatasetType.TRAIN,
        batches_path,
        cfg,
        device,
        generate_esm_embeddings,
    )
    valid_dataset = SelectionDataset(
        dataframes[1],
        DatasetType.VALIDATION,
        batches_path,
        cfg,
        device,
        generate_esm_embeddings,
    )
    test_dataset = (
        SelectionDataset(
            dataframes[2],
            DatasetType.TEST,
            batches_path,
            cfg,
            device,
            generate_esm_embeddings,
        )
        if dataframes[2] is not None
        else None
    )

    # Create DataLoaders
    dataloader_train = PhageDisplayDataLoader(
        dataset=train_dataset,
        experiment_pairs=cfg.model.splits.training_pair,
        batch_size=cfg.model.training.batch_size,
    )
    dataloader_valid = PhageDisplayDataLoader(
        dataset=valid_dataset,
        experiment_pairs=cfg.model.splits.validation_pair,
        batch_size=cfg.model.training.batch_size,
    )
    dataloader_test = (
        PhageDisplayDataLoader(
            dataset=test_dataset,
            experiment_pairs=cfg.model.splits.test_pair,
            batch_size=cfg.model.training.batch_size,
        )
        if test_dataset is not None
        else None
    )

    return (
        train_dataset,
        valid_dataset,
        test_dataset,
        dataloader_train,
        dataloader_valid,
        dataloader_test,
    )


def create_dataloader(
    cfg: DictConfig,
    train_dataset: SelectionDataset,
    valid_dataset: SelectionDataset,
    test_dataset: SelectionDataset,
):
    """Create the dataloaders for the training and validation datasets.
    If the dataloader type is "lazy", the dataloader will be a lazy dataloader
    (i.e. it will not load the entire dataset into memory, just indices,
    crucial for multiprocessing).
    If the dataloader type is "normal", the dataloader will be a normal dataloader.
    (quicker if you have enough VRAM)

    Parameters
    ----------
    cfg: DictConfig
        configuration object
    train_dataset: SelectionDataset
        training dataset
    valid_dataset: SelectionDataset
        validation dataset
    test_dataset: SelectionDataset
        test dataset

    Returns
    -------
    tuple[PhageDisplayDataLoader, PhageDisplayDataLoader, PhageDisplayDataLoader]
        tuple containing the training, validation and test dataloaders
    """
    match cfg.preprocessing.dataloader_type:
        case "lazy":
            lazy_dataloader_train = LazyPhageDisplayDataLoader(
                dataset=train_dataset,
                experiment_pairs=cfg.model.splits.training_pair,
                batch_size=cfg.model.training.batch_size,
            )
            lazy_dataloader_valid = LazyPhageDisplayDataLoader(
                dataset=valid_dataset,
                experiment_pairs=cfg.model.splits.validation_pair,
                batch_size=cfg.model.training.batch_size,
                shuffle=cfg.model.training.shuffle,
            )
            dataloader_train = torch.utils.data.DataLoader(
                lazy_dataloader_train,
                batch_size=None,
                num_workers=cfg.model.training.num_workers,
                persistent_workers=cfg.model.training.num_workers > 0,
                collate_fn=collate_and_move_to_cuda
                if cfg.preprocessing.storage_device == "cpu"
                else None,
            )
            dataloader_valid = torch.utils.data.DataLoader(
                lazy_dataloader_valid,
                batch_size=None,
                num_workers=cfg.model.training.num_workers,
                persistent_workers=cfg.model.training.num_workers > 0,
                collate_fn=collate_and_move_to_cuda
                if cfg.preprocessing.storage_device == "cpu"
                else None,
            )
            if test_dataset is not None:
                lazy_dataloader_test = LazyPhageDisplayDataLoader(
                    dataset=test_dataset,
                    experiment_pairs=cfg.model.splits.test_pair,
                    batch_size=cfg.model.training.batch_size,
                    shuffle=False,
                )
                dataloader_test = torch.utils.data.DataLoader(
                    lazy_dataloader_test,
                    batch_size=None,
                    num_workers=cfg.model.training.num_workers,
                    persistent_workers=cfg.model.training.num_workers > 0,
                    collate_fn=collate_and_move_to_cuda
                    if cfg.preprocessing.storage_device == "cpu"
                    else None,
                )
            else:
                dataloader_test = None
        case "normal":
            dataloader_train = PhageDisplayDataLoader(
                dataset=train_dataset,
                experiment_pairs=cfg.model.splits.training_pair,
                batch_size=cfg.model.training.batch_size,
            )
            dataloader_valid = PhageDisplayDataLoader(
                dataset=valid_dataset,
                experiment_pairs=cfg.model.splits.validation_pair,
                batch_size=cfg.model.training.batch_size,
                shuffle=cfg.model.training.shuffle,
            )
            if test_dataset is not None:
                dataloader_test = PhageDisplayDataLoader(
                    dataset=test_dataset,
                    experiment_pairs=cfg.model.splits.test_pair,
                    batch_size=cfg.model.training.batch_size,
                    shuffle=False,
                )
            else:
                dataloader_test = None
        case _:
            raise ValueError(
                f"Invalid dataloader type: {cfg.preprocessing.dataloader_type}"
            )

    return dataloader_train, dataloader_valid, dataloader_test


def create_datasets(
    cfg: DictConfig,
    device: torch.device,
    batches_path: Path,
    dataframes: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
    generate_esm_embeddings: bool,
):
    """Create the datasets for the training and validation datasets.
    If the test dataframe is not None, the test dataset will be created.
    
    Parameters
    ----------
    cfg: DictConfig
        configuration object
    device: torch.device
        device to use for the dataset
    batches_path: Path
        path to the batches directory
    dataframes: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        tuple containing the training dataframe, validation dataframe,
        and test dataframe.
    generate_esm_embeddings: bool
        whether to generate ESM embeddings
    
    Returns
    -------
    tuple[SelectionDataset, SelectionDataset, SelectionDataset]
        tuple containing the training dataset, validation dataset, and 
        optional test dataset.
    """
    train_dataset = SelectionDataset(
        dataframes[0],
        DatasetType.TRAIN,
        batches_path,
        cfg,
        device,
        generate_esm_embeddings,
    )
    valid_dataset = SelectionDataset(
        dataframes[1],
        DatasetType.VALIDATION,
        batches_path,
        cfg,
        device,
        generate_esm_embeddings,
    )

    if dataframes[2] is not None:
        test_dataset = SelectionDataset(
            dataframes[2],
            DatasetType.TEST,
            batches_path,
            cfg,
            device,
            generate_esm_embeddings,
        )
    else:
        test_dataset = None

    return train_dataset, valid_dataset, test_dataset


def correct_counts(selection_df: pd.DataFrame) -> pd.DataFrame:
    """
    Make sure the dataset will work with the model even in the case of a subsample,
    correcting counts per experiment.

    Parameters:
    ----------
    selection_df : pd.DataFrame
        DataFrame containing selection data, expected to have 'experiment_name',
        'initial_count', 'selected_count', 'initial_total', 'selected_total' columns.

    Returns:
    -------
    pd.DataFrame
        The DataFrame with corrected 'initial_total' and 'selected_total' columns.
    """
    # Create a copy to avoid SettingWithCopyWarning and modify original if needed
    df_corrected = selection_df.copy()

    # Iterate over each unique experiment
    for exp_name, group_df in df_corrected.groupby("experiment"):
        current_initial_total = group_df["initial_total"].iloc[0]
        calculated_initial_total = group_df["initial_count"].sum()

        if current_initial_total != calculated_initial_total:
            print(
                f"WARNING: For experiment '{exp_name}', "
                f"initial_total ({current_initial_total}) "
                f"does not match sum of initial_count ({calculated_initial_total}). "
                "Forcing initial_total to be sum of initial_count..."
            )
            df_corrected.loc[
                df_corrected["experiment"] == exp_name, "initial_total"
            ] = calculated_initial_total

        current_selected_total = group_df["selected_total"].iloc[0]
        calculated_selected_total = group_df["selected_count"].sum()

        if current_selected_total != calculated_selected_total:
            print(
                f"WARNING: For experiment '{exp_name}', "
                f"selected_total ({current_selected_total}) "
                f"does not match sum of selected_count ({calculated_selected_total}). "
                "Forcing selected_total to be sum of selected_count..."
            )
            df_corrected.loc[
                df_corrected["experiment"] == exp_name, "selected_total"
            ] = calculated_selected_total

    return df_corrected


def handle_initial_null_counts(selection_df, mode):
    """Delete initial counts that are null"""
    match mode:
        case "delete":
            selection_df = selection_df[selection_df["initial_count"] != 0]
        case "add_pseudo_counts":
            mask_initial_count_zero = selection_df["initial_count"] == 0
            selection_df.loc[
                mask_initial_count_zero, "initial_count"
            ] = selection_df.loc[mask_initial_count_zero, "selected_count"]
        case _:
            pass
    return selection_df


def adding_columns_to_df_for_pd_prediction(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Make sure the dataset contains the needed columns for the model to work."""
    columns_to_modify = [
        "initial_count",
        "selected_count",
        "initial_total",
        "selected_total",
        "experiment",
    ]
    if np.any([col in dataframe.columns for col in columns_to_modify]):
        if np.all([col in dataframe.columns for col in columns_to_modify]):
            return dataframe
        raise NonCompliantDataframeError(
            """Only some of the columns_to_modify are present in your dataset which
            is problematic.Please ensure that either all the needed columns are here
            , or none of them."""
        )
    intermediary_dataframe = pd.DataFrame(
        [[1, 1, 1, 1, "place_holder_1"] for _ in range(dataframe.shape[0])],
        columns=columns_to_modify,
    )

    return pd.concat([dataframe, intermediary_dataframe], axis=1)


def handle_kd_dataframe(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Handle the KD dataframe.
    rename the columns to match the PD dataframe.
    """
    dataframe.rename(
        columns={"VHH Sequence (AA)": "sequence", "KD (M)": "GT_Binding_Probability"},
        inplace=True,
    )
    dataframe["initial_count"] = 1
    dataframe["initial_total"] = 1
    dataframe["selected_total"] = 1

    return dataframe
