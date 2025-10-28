"""
Shape Suffixes Documentation:

S: sequences (number of sequences in the experiment)
L: sequence length (length of the input sequence)
E: embedding size (size of the embedding dimension)
F: features (hidden dimensions in MLP layers)
O: output (final output dimension)

Examples:
- sequence_embeddings_SLE: sequence embeddings with sequences, length, and
  embedding dimensions
- binding_probability_S: binding probability for each sequence
- expected_counts_S: expected counts for each sequence
- abundance_S: abundance for each sequence
"""

import pyro
import pyro.distributions as dist
import torch
from omegaconf import DictConfig
from pyro.nn import PyroModule
from torch.nn.functional import sigmoid

from src.models.constants import MODEL_CONFIG_DICT, NUMERICAL_STABILITY_CONSTANT
from src.utils.config_utils import get_model_config_dict
from src.utils.device import select_device


def _get_fitness_predictor_baseline(
    cfg: DictConfig, embedding_size: tuple[int, int], device: torch.device
) -> PyroModule:
    """
    Select chosen architecture and instantiate.
    """
    architecture = cfg.model.fitness_predictor_architecture
    config_dict = get_model_config_dict(cfg, architecture, embedding_size, device)
    config = MODEL_CONFIG_DICT[architecture]["config"](**config_dict)
    return MODEL_CONFIG_DICT[architecture]["model"](config)


class PhageDisplayPoissonRegressor(PyroModule):
    """A class using a Bayesian MLP with Poisson sampling / likelihood that predicts
    binding affinity to match / produce desired sequence abundance.
    This implementation correctly handles mini-batch training by estimating
    global normalization constants from batch-level statistics.
    """

    def __init__(
        self,
        cfg: DictConfig,
        embedding_size: tuple[int, int],
        is_null_model: bool = False,
    ):
        """
        Initializes the PhageDisplayPoissonRegressor.

        Parameters
        ----------
        cfg : DictConfig
            The configuration dictionary.
        embedding_size : tuple[int, int]
            The size of the embedding.
        """
        super().__init__()
        self._device = select_device(cfg)
        self._sequence_affinity_predictor = _get_fitness_predictor_baseline(
            cfg, embedding_size, self._device
        )
        self.cfg_dict = {
            "phage_initial_population_size": cfg.model.phage_initial_population_size,
            "estimate_method": cfg.model.training.estimate_method,
            "number_of_particles": cfg.model.training.number_of_particles,
            "model_type": cfg.model.fitness_predictor_architecture,
            "EMA_beta": cfg.model.training.EMA_beta,
        }
        self.batch_approach = {
            "cached_intermediate_sum": {},
            "list_batch_survivor_counts": [],
            "list_moving_average": {},
            "cached_EMA_sum": {},
        }
        self.is_null_model = is_null_model

    def _calculate_binding_probability(
        self, sequence_embeddings_SLE: torch.Tensor
    ) -> torch.Tensor:
        """
        Isolates the Incubation phase by predicting an affinity score from
        embeddings and converting it to a binding probability for each sequence.

        Parameters
        ----------
        sequence_embeddings_SLE : torch.Tensor
            The embeddings for each sequence with shape
            (sequences, length, embedding_size).

        Returns
        -------
        torch.Tensor
            The binding probability for each sequence with shape (sequences).
        """
        if self.is_null_model:
            # return 0 in first place because it is for
            # (1-negative_binding_probability) = 1
            return 0, torch.ones(sequence_embeddings_SLE.shape[0])

        if self._device is not None:
            sequence_embeddings_SLE = sequence_embeddings_SLE.to(self._device)

        if self.cfg_dict["model_type"] == "DoubleCNNWithNegative":
            return self._calculate_binding_probability_negative_selection(
                sequence_embeddings_SLE
            )
        if self.cfg_dict["model_type"] == "CNNWithBoltzmann":
            return self._calculate_binding_probability_boltzmann(
                sequence_embeddings_SLE
            )

        return self._calculate_binding_probability_classic(sequence_embeddings_SLE)

    def _calculate_binding_probability_classic(
        self, sequence_embeddings_SLE: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Classic way to calculate binding probability, for MLP and CNN, with just one
        probability.
        Output length = 1

        Parameters
        ----------
        sequence_embeddings_SLE : torch.Tensor
            The embeddings for each sequence with shape (sequences, length, embedding).

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            The negative binding probability (equal to 0 because we do not take it
            into account) and the binding probability with shape (sequences).
        """
        raw_output_S = self._sequence_affinity_predictor(sequence_embeddings_SLE)
        # Clamp raw output to prevent extreme values
        raw_output_S = torch.clamp(raw_output_S, min=-10.0, max=10.0)

        binding_probability_S = torch.clamp(
            sigmoid(raw_output_S),
            min=NUMERICAL_STABILITY_CONSTANT,
            max=1.0 - NUMERICAL_STABILITY_CONSTANT,
        )

        # Replace any NaN values with small positive values
        binding_probability_S = torch.nan_to_num(
            binding_probability_S, nan=NUMERICAL_STABILITY_CONSTANT
        )
        # return 0 in first place because it is for
        # (1-negative_binding_probability) = 1
        return 0, binding_probability_S

    def _calculate_binding_probability_negative_selection(
        self, sequence_embeddings_SLE: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Double tower way to calculate binding probability, for DoubleCNNWithNegative,
        with two probabilities, to simulate negative selection and incubation phase.
        Output length = 2

        Parameters
        ----------
        sequence_embeddings_SLE : torch.Tensor
            The embeddings for each sequence with shape (sequences, length, embedding).

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            The negative binding probability and the binding probability with
            shape (sequences).
        """
        (
            negative_binding_probability_S,
            binding_probability_S,
        ) = self._sequence_affinity_predictor(sequence_embeddings_SLE)

        # Clamp outputs to prevent NaN and ensure valid range
        binding_probability_S = torch.clamp(binding_probability_S, min=-10.0, max=10.0)
        negative_binding_probability_S = torch.clamp(
            negative_binding_probability_S, min=-10.0, max=10.0
        )

        binding_probability_S = torch.clamp(
            sigmoid(binding_probability_S),
            min=NUMERICAL_STABILITY_CONSTANT,
            max=1.0 - NUMERICAL_STABILITY_CONSTANT,
        )

        negative_binding_probability_S = torch.clamp(
            sigmoid(negative_binding_probability_S),
            min=NUMERICAL_STABILITY_CONSTANT,
            max=1.0 - NUMERICAL_STABILITY_CONSTANT,
        )

        # Replace any NaN values with small positive values
        binding_probability_S = torch.nan_to_num(
            binding_probability_S, nan=NUMERICAL_STABILITY_CONSTANT
        )
        negative_binding_probability_S = torch.nan_to_num(
            negative_binding_probability_S, nan=NUMERICAL_STABILITY_CONSTANT
        )

        return negative_binding_probability_S, binding_probability_S

    def _calculate_binding_probability_boltzmann(
        self, sequence_embeddings_SLE: torch.Tensor
    ) -> torch.Tensor:
        """
        Way to calculate binding probability with Boltzmann law, for CNNWithBoltzmann,
        with four probabilities, to simulate negative selection, the incubation phase,
        the target selection, and the non-target selection.
        Output length = 4

        Parameters
        ----------
        sequence_embeddings_BLC : torch.Tensor
            The embeddings for each sequence with shape (batch, length, channels).

        Returns
        -------
        torch.Tensor
            The binding probability for each sequence with shape (batch, sequences).
        """
        (
            p_binding_base_S,
            p_non_binding_base_S,
            p_binding_target_S,
            p_non_binding_target_S,
        ) = self._sequence_affinity_predictor(sequence_embeddings_SLE)

        # Clamp outputs to prevent NaN and ensure valid range
        p_binding_base_S = torch.clamp(p_binding_base_S, min=-10.0, max=10.0)
        p_non_binding_base_S = torch.clamp(p_non_binding_base_S, min=-10.0, max=10.0)
        p_binding_target_S = torch.clamp(p_binding_target_S, min=-10.0, max=10.0)
        p_non_binding_target_S = torch.clamp(
            p_non_binding_target_S, min=-10.0, max=10.0
        )

        negative_binding_probability_S = torch.clamp(
            torch.exp(-p_binding_base_S)
            / (torch.exp(-p_binding_base_S) + torch.exp(-p_non_binding_base_S)),
            min=NUMERICAL_STABILITY_CONSTANT,
            max=1.0 - NUMERICAL_STABILITY_CONSTANT,
        )

        positive_binding_probability_S = torch.clamp(
            (torch.exp(-p_binding_target_S) + torch.exp(-p_binding_base_S))
            / (
                torch.exp(-p_binding_target_S)
                + torch.exp(-p_non_binding_target_S)
                + torch.exp(-p_binding_base_S)
                + torch.exp(-p_non_binding_base_S)
            ),
            min=NUMERICAL_STABILITY_CONSTANT,
            max=1.0 - NUMERICAL_STABILITY_CONSTANT,
        )

        target_binding_probability_S = torch.clamp(
            torch.exp(-p_binding_target_S)
            / (
                torch.exp(-p_binding_target_S)
                + torch.exp(-p_non_binding_target_S)
                + torch.exp(-p_binding_base_S)
                + torch.exp(-p_non_binding_base_S)
            ),
            min=NUMERICAL_STABILITY_CONSTANT,
            max=1.0 - NUMERICAL_STABILITY_CONSTANT,
        )

        pyro.deterministic("target_binding_probability", target_binding_probability_S)

        # Replace any NaN values with small positive values
        negative_binding_probability_S = torch.nan_to_num(
            negative_binding_probability_S, nan=NUMERICAL_STABILITY_CONSTANT
        )
        positive_binding_probability_S = torch.nan_to_num(
            positive_binding_probability_S, nan=NUMERICAL_STABILITY_CONSTANT
        )

        return negative_binding_probability_S, positive_binding_probability_S

    def _calculate_expected_counts(
        self,
        binding_probability_S: torch.Tensor,
        negative_binding_probability_S: torch.Tensor,
        data: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Models the Washing, Elution, and Sequencing phases to calculate the
        final expected counts for the Poisson distribution.

        Parameters
        ----------
        binding_probability_BS : torch.Tensor
            The binding probability for each sequence with shape (batch, sequences).
        negative_binding_probability_BS : torch.Tensor
            The negative binding probability for each sequence with shape
            (batch, sequences).
        data : dict[str, torch.Tensor]
            The data dictionary containing the initial count, initial total,
            post-selection total, and total sequences in the experiment.

        Returns
        -------
        torch.Tensor
            The expected counts for each sequence with shape (sequences,).
        """
        # Load data and move to device
        initial_count_S = data["initial_count"]
        initial_total_S = data["initial_total"]
        post_selection_total_S = data["selected_total"]
        total_sequences_in_library = data["total_sequences_in_experiment"]
        experiment_name = data["experiment_name"][0]

        if self._device is not None:
            initial_count_S = initial_count_S.to(self._device, non_blocking=True)
            initial_total_S = initial_total_S.to(self._device, non_blocking=True)
            post_selection_total_S = post_selection_total_S.to(
                self._device, non_blocking=True
            )
            binding_probability_S = binding_probability_S.to(
                self._device, non_blocking=True
            )
            if isinstance(negative_binding_probability_S, torch.Tensor):
                negative_binding_probability_S = negative_binding_probability_S.to(
                    self._device, non_blocking=True
                )

        for key, value in self.cfg_dict.items():
            if isinstance(value, torch.Tensor):
                self.cfg_dict[key] = value.to(self._device, non_blocking=True)

        # Clamp probabilities to ensure valid range
        binding_probability_S = torch.clamp(
            binding_probability_S,
            min=NUMERICAL_STABILITY_CONSTANT,
            max=1.0 - NUMERICAL_STABILITY_CONSTANT,
        )
        if isinstance(negative_binding_probability_S, torch.Tensor):
            negative_binding_probability_S = torch.clamp(
                negative_binding_probability_S,
                min=NUMERICAL_STABILITY_CONSTANT,
                max=1.0 - NUMERICAL_STABILITY_CONSTANT,
            )

        # Ensure initial_total is not zero to avoid division by zero
        initial_total_S = torch.clamp(initial_total_S, min=1.0)

        survivor_counts_before_neg_S = (
            initial_count_S
            * (1 - negative_binding_probability_S)
            * (self.cfg_dict["phage_initial_population_size"] / initial_total_S)
        )

        survivor_counts_S = survivor_counts_before_neg_S * binding_probability_S

        # Replace any NaN values and ensure positive values
        survivor_counts_S = torch.nan_to_num(
            survivor_counts_S, nan=NUMERICAL_STABILITY_CONSTANT
        )
        survivor_counts_S = torch.clamp(
            survivor_counts_S, min=NUMERICAL_STABILITY_CONSTANT
        )

        if experiment_name not in self.batch_approach["list_moving_average"]:
            self.batch_approach["list_moving_average"][experiment_name] = []

        if experiment_name not in self.batch_approach["cached_EMA_sum"]:
            self.batch_approach["cached_EMA_sum"][experiment_name] = None

        if experiment_name not in self.batch_approach["cached_intermediate_sum"]:
            self.batch_approach["cached_intermediate_sum"][
                experiment_name
            ] = torch.tensor(0.0, dtype=torch.float32, device=self._device)

        if (
            self.cfg_dict["estimate_method"] == "N-1"
            or self.cfg_dict["estimate_method"] == "moving_average"
        ):
            # store detached scalar to avoid keeping computation graph across batches
            self.batch_approach["list_batch_survivor_counts"].append(
                torch.sum(survivor_counts_S).detach()
            )
        # Estimate the total "stickiness" of the enriched library post-selection.
        # It is a rough estimate of the total number of sequences in the enriched
        # library.

        intermediary_sum_estimated_global = self._estimate_intermediate_sum(
            survivor_counts_S,
            total_sequences_in_library,
            experiment_name,
        )

        # Ensure the intermediary sum is not zero to avoid division by zero
        intermediary_sum_estimated_global = torch.clamp(
            intermediary_sum_estimated_global, min=NUMERICAL_STABILITY_CONSTANT
        )

        # Simulate Elution and Sequencing: normalize by the total enriched pool
        # and scale by the sequencing depth to get the final expected counts.
        final_expected_counts_S = survivor_counts_S * (
            post_selection_total_S / intermediary_sum_estimated_global
        )

        # Final protection against NaN and ensure positive values
        final_expected_counts_S = torch.nan_to_num(
            final_expected_counts_S, nan=NUMERICAL_STABILITY_CONSTANT
        )
        final_expected_counts_S = torch.clamp(
            final_expected_counts_S, min=NUMERICAL_STABILITY_CONSTANT
        )

        return final_expected_counts_S.squeeze()

    def _sample_counts(
        self, expected_counts_S: torch.Tensor, data: dict[str, torch.Tensor]
    ) -> None:
        """
        Performs statistical inference by sampling from the Poisson distribution,
        comparing the model's predicted counts (expected_counts) to the
        real data (selected_count).

        Parameters
        ----------
        expected_counts_S : torch.Tensor
            The expected counts for each sequence with shape (sequences).
        data : dict[str, torch.Tensor]
            The data dictionary
        """
        # Final safety check: ensure expected_counts are valid for Poisson distribution
        expected_counts_S = torch.nan_to_num(
            expected_counts_S, nan=NUMERICAL_STABILITY_CONSTANT
        )
        expected_counts_S = torch.clamp(
            expected_counts_S, min=NUMERICAL_STABILITY_CONSTANT
        )

        total_sequences = data.get(
            "total_sequences_in_experiment", expected_counts_S.shape[0]
        )
        num_sequences = expected_counts_S.shape[0]

        if total_sequences == num_sequences:
            plate_ctx = pyro.plate("sequences", size=num_sequences)
        else:
            plate_ctx = pyro.plate(
                "sequences", size=total_sequences, subsample_size=num_sequences
            )

        with plate_ctx:
            post_selection_count_S = data.get("selected_count")
            if post_selection_count_S is not None:
                if self._device is not None:
                    post_selection_count_S = post_selection_count_S.to(self._device)
                pyro.sample(
                    "obs",
                    dist.Poisson(expected_counts_S + NUMERICAL_STABILITY_CONSTANT),
                    obs=post_selection_count_S.squeeze(),
                )
            else:
                pyro.sample(
                    "obs",
                    dist.Poisson(expected_counts_S + NUMERICAL_STABILITY_CONSTANT),
                )

    def _estimate_intermediate_sum(
        self,
        survivor_counts: torch.Tensor,
        total_sequences_in_library: float,
        experiment_name: str,
    ) -> torch.Tensor:
        """
        Estimates the intermediate sum for the model using cached values between epochs.
        """

        def _batch_estimate(
            survivor_counts: torch.Tensor, total_sequences_in_library: float
        ) -> torch.Tensor:
            """
            Estimates the intermediate sum for the model using cached values between
            epochs.
            """
            intermediary_sum_batch = torch.sum(survivor_counts)
            num_sequences_in_batch = survivor_counts.shape[0]

            scaling_factor = total_sequences_in_library / num_sequences_in_batch
            result = intermediary_sum_batch * scaling_factor
            return result

        # Ensure survivor_counts are valid
        survivor_counts = torch.nan_to_num(
            survivor_counts, nan=NUMERICAL_STABILITY_CONSTANT
        )
        survivor_counts = torch.clamp(survivor_counts, min=NUMERICAL_STABILITY_CONSTANT)

        match self.cfg_dict["estimate_method"]:
            case "batch":
                result = _batch_estimate(survivor_counts, total_sequences_in_library)
            case "global":
                result = torch.sum(survivor_counts)
            case "N-1":
                # Use cached value from previous epoch,
                # fallback to initial_total if not available
                if self.batch_approach["cached_intermediate_sum"][experiment_name] != 0:
                    result = self.batch_approach["cached_intermediate_sum"][
                        experiment_name
                    ]
                else:
                    result = _batch_estimate(
                        survivor_counts, total_sequences_in_library
                    )
            case "moving_average":
                if (
                    len(self.batch_approach["list_moving_average"][experiment_name])
                    != 0
                ):
                    result = torch.mean(
                        torch.stack(
                            self.batch_approach["list_moving_average"][experiment_name]
                        )
                    )
                else:
                    result = _batch_estimate(
                        survivor_counts, total_sequences_in_library
                    )
            case "EMA":
                if self.batch_approach["cached_EMA_sum"][experiment_name] is not None:
                    beta = self.cfg_dict["EMA_beta"]
                    with torch.no_grad():
                        self.batch_approach["cached_EMA_sum"][experiment_name] = (
                            beta
                            * self.batch_approach["cached_EMA_sum"][
                                experiment_name
                            ].detach()
                            + (1 - beta)
                            * _batch_estimate(
                                survivor_counts, total_sequences_in_library
                            ).detach()
                        )
                else:
                    with torch.no_grad():
                        self.batch_approach["cached_EMA_sum"][
                            experiment_name
                        ] = _batch_estimate(
                            survivor_counts, total_sequences_in_library
                        ).detach()
                result = self.batch_approach["cached_EMA_sum"][experiment_name]
            case _:
                raise ValueError(
                    f"Invalid estimate method: {self.cfg_dict['estimate_method']}"
                )

        # Ensure result is valid
        result = torch.nan_to_num(result, nan=NUMERICAL_STABILITY_CONSTANT)
        result = torch.clamp(result, min=NUMERICAL_STABILITY_CONSTANT)

        return result

    def update_intermediate_sum_cache(self, experiment_name: str) -> None:
        """
        Updates the cached intermediate sum for use in the next epoch.
        This should be called at the end of each epoch.
        """
        if self.cfg_dict["estimate_method"] == "N-1":
            with torch.no_grad():
                mean_survivor_counts = (
                    torch.sum(
                        torch.stack(self.batch_approach["list_batch_survivor_counts"])
                    )
                    / self.cfg_dict["number_of_particles"]
                ).detach()
            print(f"Updating cache with {mean_survivor_counts} survivor counts")
            self.batch_approach["cached_intermediate_sum"][
                experiment_name
            ] = mean_survivor_counts
            self.batch_approach["list_batch_survivor_counts"] = []
        elif self.cfg_dict["estimate_method"] == "moving_average":
            with torch.no_grad():
                self.batch_approach["list_moving_average"][experiment_name].append(
                    torch.sum(
                        torch.stack(self.batch_approach["list_batch_survivor_counts"])
                    ).detach()
                )
            self.batch_approach["list_batch_survivor_counts"] = []

    def forward(
        self,
        data: dict[str, torch.Tensor],
        predict: bool = False,
    ) -> None:
        """
        Executes the full forward pass of the model, simulating the phage
        display experiment to predict sequence counts.

        Parameters
        ----------
        data : dict[str, torch.Tensor]
            The data dictionary containing the embeddings, initial count,
            initial total, post-selection total, and total sequences in the experiment.
        predict : bool
            Whether to predict the counts.
        """
        sequence_embeddings_SLE = data["embeddings"]

        (
            negative_binding_probability_S,
            binding_probability_S,
        ) = self._calculate_binding_probability(sequence_embeddings_SLE)

        pyro.deterministic("binding_probability", binding_probability_S)
        if isinstance(negative_binding_probability_S, torch.Tensor):
            pyro.deterministic(
                "negative_binding_probability", negative_binding_probability_S
            )
        else:
            pyro.deterministic(
                "negative_binding_probability", torch.zeros_like(binding_probability_S)
            )

        if not predict:
            expected_counts_S = self._calculate_expected_counts(
                binding_probability_S, negative_binding_probability_S, data
            )
            self._sample_counts(expected_counts_S, data)

    def init_sum_list_to_zero(self) -> None:
        """
        Initializes the cached intermediate sum to zero.
        """
        self.batch_approach["list_batch_survivor_counts"] = []


class PhageDisplayMultinomialRegressor(PhageDisplayPoissonRegressor):
    """A class using a Bayesian MLP with multinomial sampling / likelihood that predicts
    binding affinity to match / produce desired sequence abundance.
    This implementation inherits from the baseline regressor but uses multinomial
    distribution.
    """

    def _sample_counts(
        self, expected_counts_S: torch.Tensor, data: dict[str, torch.Tensor]
    ) -> None:
        """
        Performs statistical inference by sampling from the multinomial distribution,
        comparing the model's predicted abundance to the real data (selected_count).

        Parameters
        ----------
        expected_counts_BS : torch.Tensor
            The expected counts for each sequence with shape (batch, sequences).
        data : dict[str, torch.Tensor]
            The data dictionary containing the selected count and total.
        """
        # Calculate abundance from expected counts
        abundance_S = expected_counts_S / torch.sum(expected_counts_S)

        post_selection_count_S = data.get("selected_count")
        post_selection_total_S = data["selected_total"]

        if self._device is not None:
            post_selection_total_S = post_selection_total_S.to(self._device)

        # Not sampling from the distribution (inference mode)
        if post_selection_count_S is not None:
            if self._device is not None:
                post_selection_count_S = post_selection_count_S.to(self._device)

            # Calculate likelihood given model's current parameters for gradient descent
            pyro.sample(
                "obs",
                dist.Multinomial(
                    int(post_selection_total_S[0]), abundance_S.reshape(1, -1)
                ).to_event(1),
                obs=post_selection_count_S.reshape(1, -1),
            )
        # Sampling from the distribution (prediction mode)
        else:
            pyro.sample(
                "obs",
                dist.Multinomial(
                    int(post_selection_total_S[0]),
                    abundance_S.reshape(1, -1),
                ).to_event(1),
            )
