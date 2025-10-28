# pylint: disable=R0913,R0917

from typing import Callable, Optional

import torch


class MLP(torch.nn.Sequential):
    """
    This block implements the multi-layer perceptron (MLP) module as from
    https://pytorch.org/vision/main/_modules/torchvision/ops/misc.html#MLP

    Parameters:
    ----------
    layers_size (List[int]):
        List of layers dimensions
    norm_layer (Callable[..., torch.nn.Module], optional):
        Norm layer that will be stacked on top of the linear layer. If "None" this layer
        won't be used. Default: "None".
    activation_layer (Callable[..., torch.nn.Module], optional):
        Activation function which will be stacked on top of the normalization layer (if
        not None), otherwise on top of the linear layer. If "None" this layer won't be
        used. Default: "torch.nn.ReLU".
    inplace (bool, optional):
        Parameter for the activation layer, which can also do the operation in-place.
        Default is "None", which uses the respective default values of the
        "activation_layer"" and Dropout layer.
    last_layer_has_dropout (bool):
        wether or not applying dropout to the final layer. Default: "False".
    bias (bool):
        Whether to use bias in the linear layer. Default "True".
    dropout (float):
        The probability for the dropout layer. Default: 0.0
    """

    def __init__(
        self,
        layers_size: list[int],
        norm_layer: Optional[Callable[..., torch.nn.Module]] = None,
        activation_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.ReLU,
        inplace: Optional[bool] = None,
        last_layer_has_dropout: bool = False,
        bias: bool = True,
        dropout: float = 0.0,
    ):
        """
        Constructor for the MLP.
        """
        layers = []
        for i in range(len(layers_size) - 1):
            layers.append(
                torch.nn.Linear(layers_size[i], layers_size[i + 1], bias=bias)
            )
            if i < len(layers_size) - 2:
                if norm_layer is not None:
                    layers.append(norm_layer(layers_size[i + 1]))
                layers.append(activation_layer(inplace=inplace))
                layers.append(torch.nn.Dropout(p=dropout, inplace=inplace))

        if last_layer_has_dropout:
            layers.append(torch.nn.Dropout(p=dropout, inplace=inplace))
        super().__init__(*layers)
