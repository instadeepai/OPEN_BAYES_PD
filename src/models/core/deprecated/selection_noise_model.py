# pylint: disable=not-context-manager,R0913,R0917
from typing import Optional

import pyro
import pyro.distributions as dist
import torch


class SelectionNoise(torch.nn.Module):
    """
    Given a true fitness landscape, adds knowledge driven experimental noise on top to
    recover or simulate outputs of experimental selection steps.
    """

    def __init__(self, phage_initial_population_size: int):
        """Initialize the SelectionNoise module.

        Parameters:
        -----------
        phage_initial_population_size (int):
            The initial population size of phages.
        """
        super().__init__()
        self._phage_initial_population_size = phage_initial_population_size

    def forward(
        self,
        fitness_sampled: torch.Tensor,
        pre_selection_total: torch.Tensor,
        post_selection_total: torch.Tensor,
        pre_selection_frequency: torch.Tensor,
        post_selection_count: Optional[torch.Tensor] = None,
    ):
        """
        Simulate the selection process with added experimental noise by incorporating
        sampling to simulate the noise introduced during the experimental steps.

        Parameters:
        -----------
        fitness_sampled (torch.Tensor):
            Fitness of different phage populations, each element is the fitness of a
            phage type.
        pre_selection_total (torch.Tensor):
            Total count of phages before selection.
        post_selection_total (torch.Tensor):
            Total count of phages after selection.
        pre_selection_frequency (torch.Tensor):
            Frequency of each phage type before selection, each element is the frequency
            of a phage type.
        after_selection_count (Optional[torch.Tensor]):
            Observed counts of phage types after selection. Used as observed data for
            the multinomial sampling of the post-selection counts.

        Returns:
            torch.Tensor: A tensor representing the simulated or observed counts of
            phage types after selection.
        """
        # sample initial population based on pre-selection frequencies.
        population_sampled = pyro.sample(
            "initial_sample",
            dist.Multinomial(
                int(pre_selection_total),
                probs=pre_selection_frequency.reshape(1, -1).to(torch.float64),
            ).to_event(1),
        )

        # sample selected population based on initial population and fitness.
        frequency_sampled = population_sampled / torch.sum(population_sampled)
        with pyro.poutine.scale(None, 1.0 / fitness_sampled.shape[0]):
            with pyro.plate("selected_all_plate", fitness_sampled.shape[1]):
                selected_all = pyro.sample(
                    "selected_all",
                    dist.Binomial(
                        torch.floor(
                            self._phage_initial_population_size
                            * frequency_sampled.reshape(-1, 1)
                        ),
                        probs=fitness_sampled,
                    ),
                )
        abundance = selected_all / torch.sum(selected_all)
        # sample post-selection counts based on normalized abundances.
        if post_selection_count is not None:
            obs = pyro.sample(
                "obs",
                dist.Multinomial(
                    int(post_selection_total),
                    probs=abundance.reshape(1, -1).to(torch.float64),
                ).to_event(1),
                obs=post_selection_count.reshape(1, -1).to(torch.float64),
            )
        else:
            obs = pyro.sample(
                "obs",
                dist.Multinomial(
                    int(post_selection_total),
                    probs=abundance.reshape(1, -1).to(torch.float64),
                ).to_event(1),
            )
        return obs
