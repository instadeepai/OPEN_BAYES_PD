# pylint: disable=not-context-manager,E1102
import pyro
import pyro.distributions as dist
import torch
from omegaconf import DictConfig
from torch.distributions import transforms

from src.models.core.enrichment_variance_predictor import EnrichmentVariancePredictor
from src.models.core.mean_variance_predictor import FitnessMeanVariancePredictor
from src.models.core.selection_noise_model import SelectionNoise
from src.utils.device import select_device

NUMERICAL_STABILITY_EPS: float = 1e-6


class FLIGHTEDModel(torch.nn.Module):
    """
    Implements a FLIGHTED-like model to simulate the phage display selection process
    with added experimental noise. Probabilistic model using Pyro to simulate the
    selection process in phage display experiments. It combines multiple components:
    including fitness prediction, selection noise modeling, and probabilistic
    programming, to generate realistic experimental outputs.

    Methods:
    --------
    model(data):
        Simulates the output counts of a phage display selection step, incorporating
        experimental noise.

    guide(data):
        Defines the prior for the sequence-to-fitness mapping, including noise related
        to initial and selected abundances.

    forward(sequence_embeddings):
        Returns sampled fitness values for given sequences based on the mean and
        variance predicted by the mean-variance predictor.
    """

    def __init__(self, cfg: DictConfig):
        """Initialize"""
        super().__init__()
        self._phage_initial_population_size = cfg.model.phage_initial_population_size
        self._framework = cfg.model.framework_type
        self._device = select_device(cfg)
        self._mean_variance_predictor = FitnessMeanVariancePredictor(cfg)
        self._enrichment_variance_predictor = EnrichmentVariancePredictor(cfg)
        self._selection_noise_model = SelectionNoise(
            self._phage_initial_population_size
        )
        self._mean_variance_predictor.to(self._device)
        self._enrichment_variance_predictor.to(self._device)
        self._selection_noise_model.to(self._device)

    def model(self, data: dict[str, torch.Tensor]):
        """
        Simulates output counts of a phage display selection step. Given knowledge-
        driven implementation of the selection step experimental noise, this method
        trains a noise-free fitness landscape to produce outputs that reflect the
        current understanding of the experimental noise.

        Parameters:
        -----------
        data (dict[str, torch.Tensor]):
            Input data including sequence embeddings, counts, frequencies.
        """
        pyro.module(
            "mean_variance_predictor",
            self._mean_variance_predictor,
        )

        framework_encoded = data["framework_encoded"]
        sequence_embeddings = data["embeddings"]
        if len(self._framework) > 1:  ### what is this? also in baseline_model
            sequence_embeddings = torch.hstack([sequence_embeddings, framework_encoded])

        ### frequency is not in the dataset
        pre_selection_frequency = data["initial_frequency"].to(self._device)
        pre_selection_total = data["initial_total"][0].to(self._device)  ### why [0]
        post_selection_total = data["selected_total"][0].to(self._device)  ### why [0]
        sequence_embeddings = sequence_embeddings.to(self._device)
        post_selection_count = data["selected_count"]
        if post_selection_count is not None:
            post_selection_count = data["selected_count"].to(self._device)

        fitness_mean, fitness_variance = self._mean_variance_predictor(
            sequence_embeddings
        )
        with pyro.poutine.scale(None, 1.0 / fitness_mean.shape[0]):
            with pyro.plate("fitness_draw", fitness_mean.shape[1]):
                fitness_sampled = pyro.sample(
                    "binding_probability",
                    dist.TransformedDistribution(
                        dist.Normal(fitness_mean, fitness_variance),
                        [transforms.SigmoidTransform()],
                    ),
                )

        self._selection_noise_model(
            fitness_sampled=fitness_sampled,
            pre_selection_total=pre_selection_total,
            post_selection_total=post_selection_total,
            pre_selection_frequency=pre_selection_frequency,
            post_selection_count=post_selection_count,
        )

    def guide(self, data: dict[str, torch.FloatTensor]):
        """
        Defines the prior for the sequence-to-fitness mapping. This prior indicates that
        the best guess for fitness is the enrichment ratio, acknowledging the noise
        related to initial and selected abundances. The prior distribution is Gaussian,
        centered around the enrichment and with variance predicted by an MLP.
        The model is allowed to diverge from this prior but remains coherent with it,
        through by SVI and ELBO loss training.

        Parameters:
        -----------
        data (dict[str, torch.FloatTensor]):
            Input data including sequence embeddings, counts, frequencies.
        """
        pyro.module(
            "enrichment_variance_predictor",
            self._enrichment_variance_predictor,
        )
        if data["selected_frequency"] is not None:
            pre_selection_frequency = data["initial_frequency"].to(self._device)
            pre_selection_count = data["initial_count"].to(self._device)
            post_selection_count = data["selected_count"].to(self._device)
            selectivity = data["selectivity"].to(self._device)
            pre_selection_total = data["initial_total"][0].to(self._device)
            normalized_enrichment = selectivity / torch.sum(selectivity)
            normalized_enrichment = normalized_enrichment.to(self._device)

            # Add small epsilon to avoid division by zero
            epsilon = 1e-10
            safe_pre_selection_count = torch.maximum(
                pre_selection_count, torch.tensor(epsilon).to(self._device)
            )
            safe_post_selection_count = torch.maximum(
                post_selection_count, torch.tensor(epsilon).to(self._device)
            )

            enrichment_variance = (
                1.0 / torch.sqrt(safe_pre_selection_count)
                + 1.0 / torch.sqrt(safe_post_selection_count)
            ) * normalized_enrichment
            variance_predictor_input = torch.hstack(
                [
                    normalized_enrichment,
                    enrichment_variance,
                    pre_selection_count,
                    post_selection_count,
                ]
            )
            predicted_variance = self._enrichment_variance_predictor(
                variance_predictor_input
            )
            with pyro.poutine.scale(None, 1.0):
                population_sampled = pyro.sample(
                    "initial_sample",
                    dist.Multinomial(
                        int(pre_selection_total),
                        probs=pre_selection_frequency.reshape(1, -1).to(torch.float64),
                    ).to_event(1),
                )
                frequency_sampled = population_sampled / torch.sum(population_sampled)
                with pyro.poutine.scale(None, 1.0 / normalized_enrichment.shape[0]):
                    with pyro.plate("fitness_draw", normalized_enrichment.shape[1]):
                        fitness_sampled = pyro.sample(
                            "binding_probability",
                            dist.TransformedDistribution(
                                dist.Normal(
                                    torch.special.logit(
                                        normalized_enrichment,
                                        eps=NUMERICAL_STABILITY_EPS,
                                    ),
                                    predicted_variance + NUMERICAL_STABILITY_EPS,
                                ),
                                [transforms.SigmoidTransform()],
                            ),
                        )
                    with pyro.plate("selected_all_plate", fitness_sampled.shape[1]):
                        pyro.sample(
                            "selected_all",  ### what is selected_all
                            dist.Binomial(
                                torch.floor(
                                    self._phage_initial_population_size
                                    * frequency_sampled.reshape(-1, 1)
                                ),
                                probs=fitness_sampled,
                            ),
                        )
        else:
            pass

    def forward(self, sequence_embeddings: torch.Tensor):
        """
        Returns sampled fitness values for given sequences. This method uses the mean
        and variance predicted by the mean-variance predictor to sample fitness values
        for the given sequence embeddings.

        Parameters:
        -----------
        sequence_embeddings (torch.Tensor):
            Featurized representation of sequences.

        Returns:
            torch.Tensor: Sampled fitness values.
        """
        sequence_embeddings = sequence_embeddings.to(self._device)

        fitness_mean, fitness_variance = self._mean_variance_predictor(
            sequence_embeddings
        )
        with pyro.plate("predicted_probabilities_draw", fitness_mean.shape[1]):
            predicted_fitness_sampled = pyro.sample(
                "predicted_probabilities",
                dist.TransformedDistribution(
                    dist.Normal(fitness_mean, fitness_variance),
                    [transforms.SigmoidTransform()],
                ),
            )
        return predicted_fitness_sampled
