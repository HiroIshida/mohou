from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn

from mohou.model.common import LossDict, ModelBase, ModelConfigBase


@dataclass
class LSTMConfig(ModelConfigBase):
    n_state_with_flag: int
    n_hidden: int = 200
    n_layer: int = 4
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

    def loss(self, sample: Tuple[torch.Tensor, torch.Tensor]) -> LossDict:
        state_sample, weight_seqs = sample
        state_sample_input, state_sample_output = state_sample[:, :-1], state_sample[:, 1:]
        pred_output = self.forward(state_sample_input)

        weight_seqs_expaneded = weight_seqs[:, :-1, None].expand_as(state_sample_input)
        loss_value = torch.mean(weight_seqs_expaneded * (pred_output - state_sample_output) ** 2)
        return LossDict({'prediction': loss_value})

    def forward(self, state_sample: torch.Tensor) -> torch.Tensor:
        tmp, _ = self.lstm_layer(state_sample)
        out = self.output_layer(tmp)
        return out
