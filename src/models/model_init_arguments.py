# pylint: disable=R0902
from dataclasses import dataclass
from typing import Any


@dataclass
class ResNextModelInitArguments:
    """A dataclass that defines the arguments to initialize a ResNext + downstream
    network
    """

    num_channels: list[int]
    intermediary_conv_kernel_size: list[int]
    groups: list[int]
    bot_mul: list[int]
    use_1x1conv: list[bool]
    strides: list[int]
    in_channels: int
    hidden_channels: list[int]
    last_layer_has_dropout: bool
    bias: bool
    dropout: float


def resnext_network_init_arguments_from_configs(
    config_resnext: dict[str, Any], config_downstream: dict[str, Any]
) -> ResNextModelInitArguments:
    # TODO: not used
    """Initializes a ResNext+downstream network
    given a resnext config and a mlp config
    (dictionary from the yaml files)
    """
    return ResNextModelInitArguments(**config_resnext, **config_downstream)
