import random as rand
import shutil
from pathlib import Path

import numpy as np
import pyro
import pytest
import torch

SEED_FOR_RNG: int = 42


@pytest.fixture
def _random_seeder():
    """Set random seed for all random number generators"""
    np.random.seed(SEED_FOR_RNG)
    rand.seed(SEED_FOR_RNG)
    torch.manual_seed(SEED_FOR_RNG)
    pyro.set_rng_seed(SEED_FOR_RNG)


@pytest.fixture()
def setup_and_destroy_experiment_dir():
    """Set up and destroy temporary directory for testing"""
    experiment_dir = Path(__file__).parent / "tmp_dir_experiment"
    experiment_dir.mkdir(exist_ok=False)
    yield experiment_dir
    shutil.rmtree(experiment_dir, ignore_errors=True)


def pytest_collection_modifyitems(items):
    """Modifies test items in place to ensure test modules run in a given order."""
    function_order = [
        "test_training_function_is_working_properly",
        "test_baseline_cnn_fitness_prediction_explainer",
    ]
    sorted_items = items.copy()
    final_sorted_items = []
    not_final_sorted_items = []

    for function in function_order:
        for item in sorted_items:
            if function in str(item):
                if item not in final_sorted_items:
                    final_sorted_items.append(item)
    not_final_sorted_items = list(set(sorted_items) - set(final_sorted_items))
    items[:] = final_sorted_items + not_final_sorted_items
