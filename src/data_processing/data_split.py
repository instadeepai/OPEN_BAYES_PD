import pandas as pd


def split_data(
    dataframe: pd.DataFrame, training_pairs: list[str], validation_pairs: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits the data between training and validation set according to experiment names.

    Parameters:
    -----------
    dataframe (pd.DataFrame):
        input dataset consisting in sequences, with relative counts and embeddings.
    training_pairs (list[str]):
        experiment pairs used as training data.
    validation_pairs (list[str]):
        experiment pairs used as validation data.

    Returns:
    --------
    tuple(pd.DataFrame, pd.DataFrame):
        data splits.
    """
    dataframe_train = dataframe.loc[
        dataframe["experiment"].apply(lambda x: x in training_pairs)
    ]
    dataframe_valid = dataframe.loc[
        dataframe["experiment"].apply(lambda x: x in validation_pairs)
    ]
    return dataframe_train, dataframe_valid
