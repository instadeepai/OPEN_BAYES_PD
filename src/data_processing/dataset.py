# pylint: disable=C0200,E1102,R0902,R0913,R0917,R0914
from pathlib import Path
from typing import Union

import pandas as pd
import torch
from omegaconf import DictConfig
from tqdm import tqdm

from src.data_processing.esm_embeddings import (
    batch_data,
    extract_batch_embeddings,
    load_esm_model,
)
from src.models.constants import CNN_ARCHI
from src.utils.enum_definitions import DatasetType, ModelType


# TODO: update/create dataset class to account for dataset with only sequences
class SelectionDataset:
    """Class managing the construction of the dataset of the selection experiment"""

    def __init__(
        self,
        sequence_dataframe: pd.DataFrame,
        dataset_type: DatasetType,
        batches_path: Path,
        cfg: DictConfig,
        device: torch.device,
        generate_esm_embeddings: bool = True,
    ):
        super().__init__()
        self._sequence_dataframe = sequence_dataframe
        self._dataset_type = dataset_type
        self._batches_path = batches_path
        self._cfg = cfg
        self._device = device
        self._generate_esm_embeddings = generate_esm_embeddings

        self._vhh_dictionary: dict[str, list[int | str]] = {}
        self._embeddings_generated = False

    def __len__(self):
        """Returns number of data points"""
        return len(self._sequence_dataframe)

    def __getitem__(self, item):
        """Get item"""
        # Convert tuple index to integer if necessary
        self._ensure_embeddings()
        if isinstance(item, tuple):
            item = item[0]
        experiment_name = self._vhh_dictionary["experiment_name"][item]
        experiment_round = torch.tensor(int(experiment_name[-1])).reshape(-1, 1)
        sequence = self._vhh_dictionary["sequence"][item]
        initial_count = torch.tensor(
            self._vhh_dictionary["initial_count"][item]
        ).reshape(-1, 1)
        selected_count = torch.tensor(
            self._vhh_dictionary["selected_count"][item]
        ).reshape(-1, 1)
        initial_total = torch.tensor(
            self._vhh_dictionary["initial_total"][item]
        ).reshape(-1, 1)
        selected_total = torch.tensor(
            self._vhh_dictionary["selected_total"][item]
        ).reshape(-1, 1)
        if self._cfg.model.fitness_predictor_architecture == ModelType.MLP:
            embeddings = self._vhh_dictionary["embeddings"][item].reshape(1, -1)
        elif self._cfg.model.fitness_predictor_architecture in [
            model_type.value for model_type in ModelType
        ]:
            embeddings = self._vhh_dictionary["embeddings"][item].unsqueeze(0)
        else:
            raise NotImplementedError
        initial_frequency = initial_count / initial_total
        selected_frequency = selected_count / selected_total
        selectivity = selected_frequency / initial_frequency
        return {
            "round": experiment_round,
            "experiment_name": experiment_name,
            "sequence": sequence,
            "initial_count": initial_count,
            "selected_count": selected_count,
            "initial_total": initial_total,
            "selected_total": selected_total,
            "embeddings": embeddings,
            "selectivity": selectivity,
        }

    def _convert_to_numpy(self, batch: dict[str, Union[torch.Tensor, str]]):
        # TODO: unused for now
        """Converts batches to numpy arrays to then save them as .npz files."""
        np_batch = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                np_batch[key] = value.numpy()
            else:
                np_batch[key] = value
        return np_batch

    def _ensure_embeddings(self):
        if not self._embeddings_generated and self._generate_esm_embeddings:
            self._generate_embeddings()
            self._embeddings_generated = True

    def _generate_embeddings(self):
        sequence_list = self._sequence_dataframe["sequence"].to_list()
        initial_count_list = self._sequence_dataframe["initial_count"].to_list()
        selected_count_list = self._sequence_dataframe["selected_count"].to_list()
        initial_total_list = self._sequence_dataframe["initial_total"].to_list()
        selected_total_list = self._sequence_dataframe["selected_total"].to_list()
        experiment_list = self._sequence_dataframe["experiment"].to_list()

        esm_batch_size = self._cfg.preprocessing.esm_batch_size
        fitness_predictor_architecture = self._cfg.model.fitness_predictor_architecture
        cnn_sequence_length = self._cfg.model.cnn_sequence_length
        sequence_batches = batch_data(
            [seq[: self._cfg.model.cnn_sequence_length - 2] for seq in sequence_list],
            esm_batch_size,
            False,
        )  # The -2 is after tokenization you get 2 extra tokens , one CLS and one EOS.
        # This is to ensure that the sequence length is exactly the same as the one used
        # for training the model.
        model = load_esm_model(
            self._cfg.preprocessing.esm_model_name,
            cnn_sequence_length,
            fitness_predictor_architecture,
        )
        embeddings = []
        for i in tqdm(range(len(sequence_batches))):
            embedded_batch = extract_batch_embeddings(
                model,
                sequence_batches[i],
            )
            if self._cfg.model.fitness_predictor_architecture == "MLP":
                processed_embeddings = torch.mean(embedded_batch, dim=1)
            elif self._cfg.model.fitness_predictor_architecture in CNN_ARCHI:
                processed_embeddings = embedded_batch.transpose(1, 2)
            elif self._cfg.model.fitness_predictor_architecture == "Transformer":
                processed_embeddings = embedded_batch
            else:
                raise NotImplementedError
            if self._cfg.preprocessing.storage_device == "cpu":
                embeddings.extend(processed_embeddings.cpu())
            else:
                embeddings.extend(processed_embeddings)

        self._vhh_dictionary = {
            "sequence": sequence_list,
            "embeddings": embeddings,
            "initial_count": initial_count_list,
            "selected_count": selected_count_list,
            "initial_total": initial_total_list,
            "selected_total": selected_total_list,
            "experiment_name": experiment_list,
        }

    def embedding_size(self):
        """Returns the size of the embeddings"""
        return self._vhh_dictionary["embeddings"][0].shape
