# pylint: disable=R0915,R0913,R0914,R0917,R0902

from typing import Callable, Optional

import torch
from torch import nn

from src.models.core.mlp import MLP


class ResNeXtBlock(nn.Module):
    """
    ResNeXtBlock a single ResNext layer.
    Taken from https://d2l.ai/chapter_convolutional-modern/resnet.html

    Parameters:
    ----------
    in_channels (int):
        input channel size
    out_channels (int):
        output channel size
    intermediary_conv_kernel_size (int):
        kernel size
    groups (int):
        the number by which the number of output channels will be divided and split to
        reduce computational cost
    bot_mul (int):
        an integer constant to allow the number of output channels to be divided by the
        group number
    use_1x1conv (bool):
        contains a boolean that encodes the use of a 1x1 conv kernel before the residual
        sum.
    strides (int):
        contains a int that encodes the stride size for the convolution
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        intermediary_conv_kernel_size: int,
        groups: int,
        bot_mul: int,
        use_1x1conv: bool = False,
        strides: int = 1,
    ):
        """
        Constructor for the ResNext layer.
        """
        super().__init__()
        bot_channels = int(round(out_channels * bot_mul))
        self.conv1 = nn.Conv1d(in_channels, bot_channels, kernel_size=1, stride=1)
        self.conv2 = nn.Conv1d(
            bot_channels,
            bot_channels,
            kernel_size=intermediary_conv_kernel_size,
            stride=strides,
            padding=1,
            groups=bot_channels // groups,
        )
        self.conv3 = nn.Conv1d(bot_channels, out_channels, kernel_size=1, stride=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.bn3 = nn.BatchNorm1d(out_channels)
        if use_1x1conv:  # This is used to make output compatible with skip connection
            self.conv4 = nn.Conv1d(
                in_channels, out_channels, kernel_size=1, stride=strides
            )
            self.bn4 = nn.BatchNorm1d(out_channels)
        else:
            self.conv4 = None

    def forward(self, input_data: torch.Tensor):
        """
        Forward function: run ResNeXtBlock over some input data.

        Parameters:
        ----------
        data (torch.Tensor):
            input data as embeddings of the sequence if first layer, or subsequent
            representation if not.

        Returns:
        -------
        output (arrayLike):
            sequence representation.
        """
        output_data = torch.nn.functional.relu(self.bn1(self.conv1(input_data)))
        output_data = torch.nn.functional.relu(self.bn2(self.conv2(output_data)))
        output_data = self.bn3(self.conv3(output_data))
        if self.conv4:
            input_data = self.bn4(self.conv4(input_data))
        return torch.nn.functional.relu(output_data + input_data)


class ResNeXtModel(nn.Module):
    """
    ResNextmodel consisitng of ResNext layers followed by a dowsntream MLP head.

    Parameters:
    ----------
    resnext_num_channels (list[int]):
        contains the number of channels for the whole network idx i and idx i+1 will
        create particular CNN layer
    resnext_intermediary_conv_kernel_size (list[int]):
        contains the kernel size of the 1D conv used in the whole network, where
        kernel_size idx i refers to conv1D layer i
    resnext_groups (list[int]):
        contains the group size (the number by which the number of channels will be
        divided and split to reduce computaitonal cost) of the 1D conv used in the whole
        network, where group size idx i refers to conv1D layer i
    resnext_bot_mul (list[int]):
        contains the bot_mul (an integer constant to allow the number of output channel
        to be divided by the group number) of the 1D conv used in the whole network,
        where group size idx i refers to conv1D layer i
    resnext_use_1x1conv (list[bool]):
        contains a boolean that encodes the use of a 1x1 conv kernel before the residual
        sum.
    resnext_strides (list[int]):
        contains a int that encodes the stride size for the convolution
    resnext_mlp_layers_size (list[int]):
        size of input, hidden and output layers for the downstream MLP.
    resnext_mlp_norm_layer (list[int]):
        Norm layer that will be stacked on top of the linear layer. If "None" this layer
        won't be used. Default: "None".
    resnext_mlp_inplace (list[int]):
        Parameter for the activation layer, which can also do the operation in-place.
        Default is "None", which uses the respective default values of the
        "activation_layer"" and Dropout layer.
    resnext_mlp_last_layer_has_dropout (list[int]):
        Wether or not applying dropout to the final layer. Default: "False".
    resnext_mlp_bias (list[int]):
        Whether to use bias in the linear layer. Default "True".
    resnext_mlp_dropout (list[int]):
        The probability for the dropout layer. Default: 0.0
    """

    def __init__(
        self,
        resnext_num_channels: list[int],
        resnext_intermediary_conv_kernel_size: list[int],
        resnext_groups: list[int],
        resnext_bot_mul: list[int],
        resnext_use_1x1conv: list[bool],
        resnext_strides: list[int],
        resnext_mlp_layers_size: list[int],
        resnext_mlp_norm_layer: Optional[Callable[..., torch.nn.Module]] = None,
        resnext_mlp_inplace: Optional[bool] = None,
        resnext_mlp_last_layer_has_dropout: bool = False,
        resnext_mlp_bias: bool = True,
        resnext_mlp_dropout: float = 0.0,
    ):
        """
        Constructor for the ResNext network.
        """
        # This is not great: I suggest to use a dataclass for initialization
        # of the different networks
        # def __init__(self, init_arguments: ResNextModelInitArguments):
        #  see new script dataclasses.py
        super().__init__()
        in_channels_list = resnext_num_channels[:-1]
        out_channels_list = resnext_num_channels[1:]
        resnext_layers = [
            ResNeXtBlock(
                in_c,
                out_c,
                kernel_s,
                groups,
                bot_mul,
                use1x1conv,
                strides,
            )
            for in_c, out_c, kernel_s, groups, bot_mul, use1x1conv, strides in zip(
                in_channels_list,
                out_channels_list,
                resnext_intermediary_conv_kernel_size,
                resnext_groups,
                resnext_bot_mul,
                resnext_use_1x1conv,
                resnext_strides,
            )
        ]

        self.sequence_of_resnext_block = nn.Sequential(*resnext_layers)
        self.downstream_head = MLP(
            layers_size=resnext_mlp_layers_size,
            norm_layer=resnext_mlp_norm_layer,
            inplace=resnext_mlp_inplace,
            last_layer_has_dropout=resnext_mlp_last_layer_has_dropout,
            bias=resnext_mlp_bias,
            dropout=resnext_mlp_dropout,
        )

    def forward(self, input_data: torch.Tensor):
        """
        Forward function: run ResNeXtModel over the input data.

        Parameters:
        ----------
        data (torch.Tensor):
            input data as embeddings of the sequence.

        Returns:
        -------
        output (arrayLike):
            useful sequence representation.
        """
        resnext_output = self.sequence_of_resnext_block(input_data)
        return self.downstream_head(resnext_output.reshape(resnext_output.shape[0], -1))
