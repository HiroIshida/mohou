from dataclasses import dataclass
from typing import List, Optional, Tuple, Type

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

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
    activation: Optional[str] = None  # TODO(HiroIshida): consider replace it with enum

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


@dataclass
class ProportionalModelConfig(ModelConfigBase):
    n_input: int
    n_bottleneck: int = 6
    n_layer: int = 2


class ProportionalModel(ModelBase[ProportionalModelConfig]):
    # This model is highly experimental. Maybe deleted without any notification.
    encoder: nn.Module
    decoder: nn.Module
    propagator: nn.Module
    p_value: Parameter

    def _setup_from_config(self, config: ProportionalModelConfig) -> None:
        layers = build_linear_layers(
            config.n_input, config.n_bottleneck, 100, config.n_layer, activation="tanh"
        )
        self.encoder = nn.Sequential(*layers)

        layers = build_linear_layers(
            config.n_bottleneck, config.n_input, 100, config.n_layer, activation="tanh"
        )
        self.decoder = nn.Sequential(*layers)

        param = Parameter(torch.zeros(1))
        self.register_parameter("kp", param)
        self.p_value = param

    def loss(self, sample: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> LossDict:
        # NOTE: episode index and static context is not supported in this model
        _, seq_sample, _ = sample

        n_batch, n_seq_len, n_dim = seq_sample.shape
        sample_pre = seq_sample[:, :-1, :].reshape(-1, n_dim)
        sample_post = seq_sample[:, 1:, :].reshape(-1, n_dim)

        z_pre = self.encoder(sample_pre)

        sample_pre_reconst = self.decoder(z_pre)
        z_post_est = (1.0 - self.p_value) * z_pre
        z_post = self.encoder(sample_post)

        f_loss = nn.MSELoss()
        reconstruction_loss = f_loss(sample_pre_reconst, sample_pre)
        prediction_loss = f_loss(z_post_est, z_post)
        return LossDict({"reconstruction": reconstruction_loss, "prediction": prediction_loss})

    def forward(self, seq_sample: torch.Tensor) -> torch.Tensor:
        n_batch, n_seq_len, n_dim = seq_sample.shape
        sample_pre = seq_sample.reshape(-1, n_dim)
        z = self.encoder(sample_pre)
        z_post = (1.0 - self.p_value) * z
        sample_post = self.decoder(z_post)
        return sample_post
