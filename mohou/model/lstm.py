from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn

from mohou.model.common import LossDict, ModelBase, ModelConfigBase


@dataclass
class LSTMConfig(ModelConfigBase):
    n_state_with_flag: int
    n_static_context: int = 0
    n_hidden: int = 200
    n_layer: int = 4
    n_output_layer: int = 1


class LSTM(ModelBase[LSTMConfig]):
    """
    lstm: x_t+1 = f(x_t, x_t-1, ...)
    lstm with context: x_t+1 = f(x_t, x_t-1, ..., c) where c is static (time-invariant) context
    """

    lstm_layer: nn.LSTM
    output_layer: nn.Sequential

    def _setup_from_config(self, config: LSTMConfig) -> None:
        n_state = config.n_state_with_flag
        n_ts_input = config.n_static_context
        self.lstm_layer = nn.LSTM(
            n_state + n_ts_input, config.n_hidden, config.n_layer, batch_first=True
        )
        output_layers = []
        for _ in range(config.n_output_layer):
            output_layers.append(nn.Linear(config.n_hidden, config.n_hidden))
        output_layers.append(nn.Linear(config.n_hidden, n_state))
        self.output_layer = nn.Sequential(*output_layers)

    def loss(self, sample: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> LossDict:
        state_sample, static_context_sample, weight_seqs = sample

        # sanity check
        n_batch, n_seq_len, n_dim = state_sample.shape
        assert static_context_sample.ndim == 2
        assert static_context_sample.shape[0] == n_batch
        assert static_context_sample.shape[1] == self.config.n_static_context

        assert weight_seqs.ndim == 2
        assert weight_seqs.shape[0] == n_batch
        assert weight_seqs.shape[1] == n_seq_len

        # propagation
        state_sample_input, state_sample_output = state_sample[:, :-1], state_sample[:, 1:]
        pred_output = self.forward(state_sample_input, static_context_sample)

        weight_seqs_expaneded = weight_seqs[:, :-1, None].expand_as(state_sample_input)
        loss_value = torch.mean(weight_seqs_expaneded * (pred_output - state_sample_output) ** 2)
        return LossDict({"prediction": loss_value})

    def forward(self, state_sample: torch.Tensor, static_context: torch.Tensor) -> torch.Tensor:
        # arrange bais_sample and create concat state_sample
        _, n_seq_len, _ = state_sample.shape
        context_unsqueezed = static_context.unsqueeze(dim=1)
        context_sequenced = context_unsqueezed.expand(-1, n_seq_len, -1)
        context_auged_state_sample = torch.cat((state_sample, context_sequenced), dim=2)

        # propagation
        tmp, _ = self.lstm_layer(context_auged_state_sample)
        out = self.output_layer(tmp)
        return out
