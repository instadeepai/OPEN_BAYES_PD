import functools
from typing import Any, Optional

import torch
from transformers import AutoModel, AutoTokenizer

from src.models.constants import CNN_ARCHI


@functools.cache
def load_esm_model(
    model_name: str, seq_max_length: int, fitness_predictor_architecture: str
):
    """
    Load chosen pre-trained ESM model.
    """
    return HFESMEmbedding(model_name, seq_max_length, fitness_predictor_architecture)


def batch_sequences(
    sequence_list: list[str], batch_size: int, drop_last: Optional[bool] = False
) -> list[list[str]]:
    """
    Splits a list of sequences into batches of size batch_size.

    Parameters:
    -----------
    sequence_list (list[str]):
        A list of sequences to be batched.
    batch_size (int):
        The maximum number of sequences per batch.
    drop_last (Optional[bool]):
        If True, the last batch will be dropped if it contains fewer than `batch_size`
        sequences. Defaults to False.

    Returns:
    list[list[str]]:
        A list of batches, where each batch is a list of sequences.
    """
    sequence_batch_list = []
    number_of_sequence = len(sequence_list)
    for i in range(0, number_of_sequence, batch_size):
        sequence_batch = sequence_list[i : i + batch_size]
        if drop_last and len(sequence_batch) < batch_size:
            continue
        sequence_batch_list.append(sequence_batch)
    return sequence_batch_list


def batch_data(
    input_list: list[str], batch_size: int, drop_last: Optional[bool] = False
) -> list[list[Any]]:
    """
    Splits a list of strings into batches of size batch_size.

    Parameters:
    -----------
    sequence_list (list[str]):
        A list of strings to be batched.
    batch_size (int):
        The maximum number of data point per batch.
    drop_last (Optional[bool]):
        If True, the last batch will be dropped if it contains fewer than `batch_size`
        data points. Defaults to False.

    Returns:
    list[list[Any]]:
        A list of batches, where each batch is a list of data points.
    """
    sequence_batch_list = []
    number_of_sequence = len(input_list)
    for i in range(0, number_of_sequence, batch_size):
        sequence_batch = input_list[i : i + batch_size]
        if drop_last and len(sequence_batch) < batch_size:
            continue
        sequence_batch_list.append(sequence_batch)
    return sequence_batch_list


def extract_batch_embeddings(
    model: torch.nn.Module,
    sequence_batch: list[str],
) -> torch.Tensor:
    """
    Generate ESM embeddings for a batch of amino acid sequences.

    Parameters
    ----------
    model (torch.nn.Module):
        ESM model adopted.
    sequence_batch (list[str]):
        Batch of protein sequences to process.

    Returns
    -------
    torch.Tensor:
        Embedded batch of sequences.
    """
    with torch.no_grad():
        model_output = model(
            sequence_batch,
        )
    return model_output


class HFESMEmbedding(torch.nn.Module):
    """Embeds the sequences ready to be fed to NanoBert and average the representation
    over the sequence length.
    """

    def __init__(
        self, model_name: str, seq_max_length: int, fitness_predictor_architecture: str
    ):
        super().__init__()
        self.model_name = model_name

        self.esm_model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
        )
        # this if/else statement permit to both use models
        # with tokenizer in attributes and models or in a separate object
        if hasattr(self.esm_model, "tokenizer"):
            self.tokenizer = self.esm_model.tokenizer
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.esm_model.eval()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.esm_model = self.esm_model.to(self.device)
        self.seq_max_length = seq_max_length
        self.fitness_predictor_architecture = fitness_predictor_architecture

    def forward(self, sequence: list[str]) -> torch.Tensor:
        """Embeds the sequences ready to be fed to NanoBert and average the
        representation over the sequence length.
        """
        inputs = self.tokenizer(
            sequence,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.seq_max_length,
        )
        if (
            self.fitness_predictor_architecture in CNN_ARCHI
            or self.fitness_predictor_architecture == "Transformer"
        ):
            inputs_ids = torch.hstack(
                [
                    inputs["input_ids"],
                    torch.ones(
                        (
                            inputs["input_ids"].shape[0],
                            self.seq_max_length - inputs["input_ids"].shape[1],
                        ),
                        dtype=torch.long,
                    ),
                ]
            )
            inputs_attention_mask = torch.hstack(
                [
                    inputs["attention_mask"],
                    torch.zeros(
                        (
                            inputs["attention_mask"].shape[0],
                            self.seq_max_length - inputs["attention_mask"].shape[1],
                        ),
                        dtype=torch.long,
                    ),
                ]
            )
            inputs = {
                "input_ids": inputs_ids.to(self.device),
                "attention_mask": inputs_attention_mask.to(self.device),
            }
        else:
            inputs = inputs.to(self.device)
        with torch.no_grad():
            outputs = self.esm_model(**inputs)
        # Get embeddings from the last hidden layer
        return outputs.last_hidden_state
