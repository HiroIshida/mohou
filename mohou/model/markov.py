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
    # p_value: Parameter
    p_value: nn.Module

    def _setup_from_config(self, config: ProportionalModelConfig) -> None:
        layers = build_linear_layers(
            config.n_input, config.n_bottleneck, 100, config.n_layer, activation="tanh"
        )
        self.encoder = nn.Sequential(*layers)

        layers = build_linear_layers(
            config.n_bottleneck, config.n_input, 100, config.n_layer, activation="tanh"
        )
        self.decoder = nn.Sequential(*layers)

        # param = Parameter(torch.ones(1) * 0.0)
        # self.register_parameter("kp", param)
        layers = build_linear_layers(config.n_bottleneck, 1, config.n_bottleneck * 2, 2, "tanh")
        layers.append(nn.Tanh())
        self.p_value = nn.Sequential(*layers)

    def get_p_value(self, bottoleneck_tensor: torch.Tensor) -> torch.Tensor:
        return 0.5 * (self.p_value(bottoleneck_tensor) + 0.5)

    def loss(self, sample: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> LossDict:
        # NOTE: episode index and static context is not supported in this model
        _, seq_sample, _ = sample

        n_batch, n_seq_len, n_dim = seq_sample.shape
        X0 = seq_sample.reshape(-1, n_dim)

        Z0 = self.encoder(X0)

        n_window_desired = 10
        n_window_max = Z0.shape[1] - 1
        n_window = min(n_window_desired, n_window_max)
        prediction_loss: Optional[torch.Tensor] = None
        Z_prop_est = Z0
        for window in range(1, n_window + 1):
            p_value = self.get_p_value(Z_prop_est)
            print("min: {}, max {}".format(torch.min(p_value), torch.max(p_value)))
            Z_prop_est = Z_prop_est * (1 - p_value)
            Z_prop_est_cut = Z_prop_est[:, :-window]
            Z_prop = Z0[:, window:]
            partial_loss = nn.MSELoss()(Z_prop, Z_prop_est_cut)
            if prediction_loss is None:
                prediction_loss = partial_loss
            else:
                prediction_loss += partial_loss
        prediction_loss = prediction_loss / float(n_window_desired)

        f_loss = nn.MSELoss()
        X0_hat = self.decoder(Z0)
        reconstruction_loss = f_loss(X0_hat, X0)
        assert prediction_loss is not None
        return LossDict({"reconstruction": reconstruction_loss, "prediction": prediction_loss})

    def forward(self, seq_sample: torch.Tensor) -> torch.Tensor:
        n_batch, n_seq_len, n_dim = seq_sample.shape
        sample_pre = seq_sample.reshape(-1, n_dim)
        z = self.encoder(sample_pre)
        p_value = self.get_p_value(z)
        z_post = (1.0 - p_value) * z
        sample_post = self.decoder(z_post)
        return sample_post
