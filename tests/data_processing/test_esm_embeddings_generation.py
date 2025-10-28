# pylint: disable=W0104,W0611
import os

import pytest
import torch
from hydra import compose, initialize

from src.data_processing.esm_embeddings import (
    batch_sequences,
    extract_batch_embeddings,
    load_esm_model,
)
from src.utils.device import select_device

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


@pytest.mark.usefixtures("_random_seeder")
def test_esm_embeddings_are_generated_correctly():
    """test esm embeddings generation functions work as expected"""
    with initialize(version_base=None, config_path="../config"):
        cfg = compose(config_name="default.yaml")

    model_name = cfg.preprocessing.esm_model_name
    esm_batch_size = cfg.preprocessing.esm_batch_size
    fitness_predictor_architecture = cfg.model.fitness_predictor_architecture
    cnn_sequence_length = cfg.model.cnn_sequence_length
    device = select_device(cfg)
    sequence_list = [
        "AALALALLALLALLAL",
        "AKALALALLALALA",
        "AALALALALA",
        "AALALAAALLLA",
        "ALALKKLLALLALALA",
        "AALLALLALKALA",
        "AALLALLALALA",
        "LALALAALALA",
        "LKALALLALALA",
        "LALALKALLALLALA",
    ]
    sequence_batches = batch_sequences(sequence_list, esm_batch_size, False)
    model = load_esm_model(
        model_name, cnn_sequence_length, fitness_predictor_architecture
    )

    embeddings = []
    for sequence_batch in sequence_batches:
        embedded_batch = extract_batch_embeddings(
            model,
            sequence_batch,
        )
        embeddings.append(embedded_batch)

    expected_shape = [[8, 18, 320], [2, 17, 320]]
    assert list(embeddings[0].shape) == expected_shape[0]
    assert list(embeddings[1].shape) == expected_shape[1]

    # fmt: off
    expected_batch_1_seq_3_embeddings_mean = torch.tensor([
        -0.0112, -0.0102, -0.0085, -0.0084, -0.0093, -0.0085, -0.0097, -0.0087, -0.0091,
        -0.0083, -0.0090, -0.0098, -0.0076, -0.0076, -0.0075, -0.0071, -0.0074, -0.0077,
    ]).to(device)
    expected_batch_2_seq_1_embeddings_mean = torch.tensor([
        -0.0103, -0.0091, -0.0088, -0.0085, -0.0081, -0.0085, -0.0078, -0.0081, -0.0091,
        -0.0080, -0.0088, -0.0079, -0.0084, -0.0094, -0.0068, -0.0069, -0.0069,
    ]).to(device)
    # fmt: on
    assert torch.allclose(
        torch.mean(embeddings[0], dim=2)[2],
        expected_batch_1_seq_3_embeddings_mean,
        rtol=1e-04,
        atol=1e-04,
    )
    assert torch.allclose(
        torch.mean(embeddings[1], dim=2)[0],
        expected_batch_2_seq_1_embeddings_mean,
        rtol=1e-04,
        atol=1e-04,
    )
