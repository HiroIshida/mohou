from dataclasses import dataclass

import torch
import torch.nn as nn

from mohou.model.common import LossDict, ModelBase, ModelConfigBase


@dataclass
class LSTMConfig(ModelConfigBase):
    n_state_with_flag: int
    n_hidden: int = 200
    n_layer: int = 2
    n_output_layer: int = 1


class LSTM(ModelBase[LSTMConfig]):
    lstm_layer: nn.LSTM
    output_layer: nn.Sequential

    def _setup_from_config(self, config: LSTMConfig) -> None:
        n_state = config.n_state_with_flag
        self.lstm_layer = nn.LSTM(n_state, config.n_hidden, config.n_layer, batch_first=True)
        output_layers = []
        for _ in range(config.n_output_layer):
            output_layers.append(nn.Linear(config.n_hidden, config.n_hidden))
        output_layers.append(nn.Linear(config.n_hidden, n_state))
        self.output_layer = nn.Sequential(*output_layers)

    def loss(self, sample: torch.Tensor) -> LossDict:
        sample_input, sample_output = sample[:, :-1], sample[:, 1:]
        pred_output = self.forward(sample_input)
        loss_value = nn.MSELoss()(pred_output, sample_output)
        return LossDict({'prediction': loss_value})

    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        tmp, _ = self.lstm_layer(sample)
        out = self.output_layer(tmp)
        return out
