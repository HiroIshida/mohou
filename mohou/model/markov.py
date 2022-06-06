from dataclasses import dataclass
from typing import List, Optional, Tuple, Type

import torch
import torch.nn as nn

from mohou.model.common import LossDict, ModelBase, ModelConfigBase


def build_linear_layers(
    n_input: int, n_output: int, n_hidden: int, n_layer: int, activation: Optional[str]
) -> List[nn.Module]:

    if activation is not None:
        assert activation in ("relu", "sigmoid", "tanh")

    AT: Optional[Type[nn.Module]] = None
    if activation == "relu":
        AT = nn.ReLU
    elif activation == "sigmoid":
        AT = nn.Sigmoid
    elif activation == "tanh":
        AT = nn.Tanh

    layers: List[nn.Module] = []
    input_layer = nn.Linear(n_input, n_hidden)
    layers.append(input_layer)
    if AT is not None:
        layers.append(AT())

    for _ in range(n_layer):
        middle_layer = nn.Linear(n_hidden, n_hidden)
        layers.append(middle_layer)
        if AT is not None:
            layers.append(AT())

    output_layer = nn.Linear(n_hidden, n_output)
    layers.append(output_layer)
    return layers


@dataclass
class MarkoveModelConfig(ModelConfigBase):
    n_input: int
    n_output: int
    n_hidden: int = 200
    n_layer: int = 4
    activation: Optional[str] = None

    def __post_init__(self):
        if self.activation is not None:
            assert self.activation in ("relu", "sigmoid", "tanh")


class ControlModel(ModelBase):
    layer: nn.Sequential

    def _setup_from_config(self, config: MarkoveModelConfig) -> None:
        config.n_input
        layers = build_linear_layers(
            n_input=config.n_input,
            n_output=config.n_output,
            n_hidden=config.n_hidden,
            n_layer=config.n_layer,
            activation=config.activation,
        )
        self.layer = nn.Sequential(*layers)

    def loss(self, sample: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> LossDict:
        inp_ctrl_sample, inp_obs_sample, out_obs_sample = sample
        inp_sample = torch.concat((inp_ctrl_sample, inp_obs_sample), dim=1)
        out_obs = self.layer(inp_sample)
        loss = nn.MSELoss()(out_obs_sample, out_obs)
        return LossDict({"prediction": loss})
        return self.layer(sample)
