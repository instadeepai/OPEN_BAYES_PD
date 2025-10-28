import multiprocessing as mp
from typing import Dict, List, Optional, Union

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import get_worker_info

from src.data_processing.dataset import SelectionDataset

mp.set_start_method("spawn", force=True)


class LazyPhageDisplayDataLoader(torch.utils.data.IterableDataset):
    """Custom dataloader for phage display experiments."""

    def __init__(
        self,
        dataset: SelectionDataset,
        experiment_pairs: list[str],
        batch_size: Optional[int] = None,
        shuffle: bool = True,
    ):
        self._dataset = dataset
        self._experiment_pairs = experiment_pairs
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._mapping_experiment_pairs_to_int = {
            experiment_pair: i for i, experiment_pair in enumerate(experiment_pairs)
        }
        self._experiment_indices: List[List[int]] = [
            [] for _ in range(len(self._experiment_pairs))
        ]

        # Initialize by iterating over dataset items
        print(
            "Initializing by iterating over dataset items. "
            "This can be slow for large datasets."
        )
        for i, data in enumerate(self._dataset):
            experiment_name = data.get("experiment_name")  # Use .get() for safety
            if (
                experiment_name
                and experiment_name in self._mapping_experiment_pairs_to_int
            ):
                exp_idx = self._mapping_experiment_pairs_to_int[experiment_name]
                self._experiment_indices[exp_idx].append(i)

        # Precompute batch metadata for __len__ and efficient iteration
        self._precomputed_batch_boundaries = self._calculate_all_batch_boundaries()

    def _collate_fn(
        self,
        batch_indices: List[int],
        exp_total: int,  # This is the total number of sequences in the experiment
    ) -> Dict[str, Union[Tensor, List[str]]]:
        if not batch_indices:
            # Consistent structure for empty batch
            return {
                "initial_count": torch.empty(0, 1, dtype=torch.float32),
                "selected_count": torch.empty(0, 1, dtype=torch.float32),
                "initial_total": torch.empty(0, 1, dtype=torch.float32),
                "selected_total": torch.empty(0, 1, dtype=torch.float32),
                "sequence": [],
                "experiment_name": [],
                "embeddings": torch.empty(
                    0, self._dataset[0]["embeddings"].shape[0], dtype=torch.float32
                ),
                "selectivity": torch.empty(0, 1, dtype=torch.float32),
                "total_sequences_in_experiment": exp_total,
                "subsample_indices": torch.empty(0, dtype=torch.long),
            }
        batch_data_points = [self._dataset[idx] for idx in batch_indices]
        return {
            "initial_count": torch.vstack(
                [single["initial_count"] for single in batch_data_points]
            ),
            "selected_count": torch.vstack(
                [single["selected_count"] for single in batch_data_points]
            ),
            "initial_total": torch.vstack(
                [single["initial_total"] for single in batch_data_points]
            ),
            "selected_total": torch.vstack(
                [single["selected_total"] for single in batch_data_points]
            ),
            "sequence": [single["sequence"] for single in batch_data_points],
            "experiment_name": [
                single["experiment_name"] for single in batch_data_points
            ],
            "embeddings": torch.vstack(
                [single["embeddings"] for single in batch_data_points]
            ),
            "selectivity": torch.vstack(
                [single["selectivity"] for single in batch_data_points]
            ),
            "total_sequences_in_experiment": exp_total,  # Per experiment
            "subsample_indices": torch.tensor(batch_indices, dtype=torch.long),
        }

    def _calculate_all_batch_boundaries(self) -> List[tuple[int, int, int]]:
        """
        Pre-calculates (experiment_idx, start_idx_in_exp, end_idx_in_exp) for ALL
        batches. This provides a definitive list of batch definitions.
        """
        all_batch_boundaries = []
        for exp_idx, exp_indices in enumerate(self._experiment_indices):
            if not exp_indices:
                continue

            num_sequences_in_exp = len(exp_indices)

            if self._batch_size is None:
                # Single batch per experiment
                all_batch_boundaries.append((exp_idx, 0, num_sequences_in_exp))
            else:
                # Multiple batches per experiment
                for start in range(0, num_sequences_in_exp, self._batch_size):
                    end = min(start + self._batch_size, num_sequences_in_exp)
                    all_batch_boundaries.append((exp_idx, start, end))
        return all_batch_boundaries

    def __len__(self):
        """Number of batches that will be produced."""
        return len(self._precomputed_batch_boundaries)

    def _get_worker_batch_range(self):
        """Get the start and end indices of batches for the current worker."""
        worker_info = get_worker_info()
        if worker_info is None:
            # Single-process or main process
            return 0, len(self._precomputed_batch_boundaries), 42  # Default seed

        worker_id = worker_info.id
        num_workers = worker_info.num_workers

        total_batches = len(self._precomputed_batch_boundaries)
        per_worker_batches = (total_batches + num_workers - 1) // num_workers
        batch_start_idx = worker_id * per_worker_batches
        batch_end_idx = min(batch_start_idx + per_worker_batches, total_batches)

        seed = worker_info.seed % (2**32)
        return batch_start_idx, batch_end_idx, seed

    def __iter__(self):
        """
        Iterates over the dataset, yielding batches.
        This method is now the central point for batch generation.
        """
        batch_start_idx, batch_end_idx, seed = self._get_worker_batch_range()

        # Create a local copy of experiment_indices for shuffling per worker
        # This is crucial to avoid modifying the shared state in multiprocessing
        worker_experiment_indices = [
            list(indices) for indices in self._experiment_indices
        ]

        # Apply shuffling if enabled, using the worker-specific seed
        if self._shuffle:
            rng = np.random.default_rng(seed)  # Use modern numpy random generator
            for exp_indices_list in worker_experiment_indices:
                if exp_indices_list:
                    rng.shuffle(exp_indices_list)

        # Determine the sequence of global batch indices for this worker
        # If not shuffling experiments, this is just a slice.
        # If shuffling experiments, we need a more complex strategy.
        # For simplicity, let's assume `_precomputed_batch_boundaries`
        # already represents the global order *before* per-experiment shuffle.
        # The worker's `batch_start_idx` and `batch_end_idx` refer to this global list.
        batches_for_this_worker = self._precomputed_batch_boundaries[
            batch_start_idx:batch_end_idx
        ]

        # Yield batches based on the precomputed boundaries
        for exp_idx, start_in_exp, end_in_exp in batches_for_this_worker:
            # Use the potentially shuffled indices for the current experiment
            current_exp_indices = worker_experiment_indices[exp_idx]

            # Select the batch indices from the (potentially shuffled) experiment list
            batch_data_indices = current_exp_indices[start_in_exp:end_in_exp]

            # Get the total number of sequences in this specific experiment
            exp_total_sequences = len(
                self._experiment_indices[exp_idx]
            )  # Use original length

            yield self._collate_fn(batch_data_indices, exp_total_sequences)

    def __getitem__(self, idx: int) -> Dict[str, Union[Tensor, List[str]]]:
        """
        Allows direct access to a single data point from the underlying dataset.
        This is separate from batching.
        """
        return self._dataset[idx]


class PhageDisplayDataLoader:
    """A custom dataloader for phage display experiments"""

    def __init__(
        self,
        dataset: SelectionDataset,
        experiment_pairs: list[str],
        batch_size: Optional[int] = None,  # max sequences per mini-batch
        shuffle: bool = True,  # whether to shuffle the data at each epoch
    ):
        """Init

        Parameters
        ----------
        dataset : SelectionDataset
            The dataset containing every sequence of all experiments.
        experiment_pairs : list[str]
            Names of the experiments we want to iterate over.
        batch_size : Optional[int]
            If provided, experiments are split into mini-batches of at most this
            size. If ``None`` (default) each yielded batch == full experiment.
        shuffle : bool
            Whether to shuffle the data at each epoch. Default is True.
        num_workers : Optional[int]
            Placeholder for future multiprocessing support.
        """
        self._dataset = dataset
        self._experiment_pairs = experiment_pairs
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._mapping_experiment_pairs_to_int = {
            experiment_pair: i for i, experiment_pair in enumerate(experiment_pairs)
        }

        # Group sequences by experiment once during initialization
        self._experiments = [[] for _ in range(len(self._experiment_pairs))]
        self._global_indices = [[] for _ in range(len(self._experiment_pairs))]
        dataset_items = [self._dataset[i] for i in range(len(self._dataset))]
        for i, sequence_data in enumerate(dataset_items):
            for experiment_pair in self._experiment_pairs:
                if sequence_data["experiment_name"] == experiment_pair:
                    exp_idx = self._mapping_experiment_pairs_to_int[experiment_pair]
                    self._experiments[exp_idx].append(sequence_data)
                    self._global_indices[exp_idx].append(i)
                    break

    # batch is 1 experiment, collate more sequences to construct the entire experiment
    def _collate_fn(
        self,
        batch: list[dict[str, Tensor | str]],
        exp_total: int,
    ) -> dict[str, Tensor | list[str]]:
        """Prepares a batch from a list of dictionary encoding for a datapoint.

        Parameters
        ----------
        batch : list[dict]
            sequences selected for this mini-batch
        exp_total : int
            total number of unique sequences in the full experiment (used for
            proper plate scaling)
        """
        return {
            "initial_count": torch.vstack(
                [single["initial_count"] for single in batch]
            ),
            "selected_count": torch.vstack(
                [single["selected_count"] for single in batch]
            ),
            "initial_total": torch.vstack(
                [single["initial_total"] for single in batch]
            ),
            "selected_total": torch.vstack(
                [single["selected_total"] for single in batch]
            ),
            "sequence": [single["sequence"] for single in batch],
            "experiment_name": [single["experiment_name"] for single in batch],
            "embeddings": torch.vstack([single["embeddings"] for single in batch]),
            "selectivity": torch.vstack([single["selectivity"] for single in batch]),
            "total_sequences_in_experiment": exp_total,
        }

    def __len__(self):
        """Number of batches that will be produced."""
        if self._batch_size is None:
            # One batch per non-empty experiment
            return sum(1 for exp in self._experiments if exp)

        total_batches = 0
        for experiment in self._experiments:
            if not experiment:
                continue
            count = len(experiment)
            total_batches += (count + self._batch_size - 1) // self._batch_size
        return total_batches

    def _shuffle_experiments(self, experiments):
        """Shuffle experiments if needed."""
        if not self._shuffle:
            return experiments

        for i, experiment in enumerate(experiments):
            if experiment:
                indices = np.random.permutation(len(experiment))
                experiments[i] = [experiment[j] for j in indices]
        return experiments

    def _yield_batches(self, experiments):
        """Yield batches from experiments."""
        for experiment in experiments:
            if not experiment:
                continue

            if self._batch_size is None:
                # Full experiment as a single batch
                batch_dict = self._collate_fn(experiment, len(experiment))
                yield batch_dict
            else:
                num_sequences = len(experiment)
                effective_batch_size = min(self._batch_size, num_sequences)

                for start in range(0, num_sequences, effective_batch_size):
                    end = min(start + effective_batch_size, num_sequences)
                    mini_batch = experiment[start:end]
                    batch_dict = self._collate_fn(
                        mini_batch,
                        num_sequences,
                    )
                    yield batch_dict

    def __iter__(self):
        """Simple iterator with low cognitive complexity."""
        # Create a copy of experiments to shuffle
        experiments = [exp.copy() for exp in self._experiments]

        # Shuffle each experiment's sequences if shuffle is True
        experiments = self._shuffle_experiments(experiments)

        # Yield batches
        yield from self._yield_batches(experiments)
