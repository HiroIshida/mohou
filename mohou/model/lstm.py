from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Type

import torch
import torch.nn as nn

from mohou.model.common import LossDict, ModelBase, ModelConfigBase
from mohou.types import ElementBase


@dataclass
class LSTMConfig(ModelConfigBase):
    """
    type_wise_loss: if True, loss is computed for each type (following type_bound_table) and
    each loss is stored in the LossDict

    NOTE: loss.total() with type_wise = True end False will not match in general.
    By type_wise = True, loss of each type is equaly treated regardless of its dimension
    If type_wise = False, if, say the state is composed of 1dim vector and 16dim vector
    1dim vector's relative importance is too small because, the loss will take the average
    over the loss of entire state
    """

    n_state_with_flag: int
    n_static_context: int = 0
    n_hidden: int = 200
    n_layer: int = 4
    n_output_layer: int = 1
    type_wise_loss: bool = False
    type_bound_table: Optional[Dict[Type[ElementBase], slice]] = None


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
        pred_output, _ = self.forward(state_sample_input, static_context_sample)
        weight_seqs_expaneded = weight_seqs[:, :-1, None].expand_as(state_sample_input)

        if self.config.type_wise_loss:
            assert self.config.type_bound_table is not None
            d = {}
            for elem_type, bound in self.config.type_bound_table.items():
                pred_output_partial = pred_output[:, :, bound]
                state_sample_output_partial = state_sample_output[:, :, bound]
                weight_seqs_partial = weight_seqs_expaneded[:, :, bound]
                loss_value_partial = torch.mean(
                    weight_seqs_partial * (pred_output_partial - state_sample_output_partial) ** 2
                )
                key = elem_type.__name__
                d[key] = loss_value_partial
            return LossDict(d)
        else:
            loss_value = torch.mean(
                weight_seqs_expaneded * (pred_output - state_sample_output) ** 2
            )
            return LossDict({"prediction": loss_value})

    def forward(
        self,
        state_sample: torch.Tensor,
        static_context: torch.Tensor,
        hidden: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # arrange bais_sample and create concat state_sample
        _, n_seq_len, _ = state_sample.shape
        context_unsqueezed = static_context.unsqueeze(dim=1)
        context_sequenced = context_unsqueezed.expand(-1, n_seq_len, -1)
        context_auged_state_sample = torch.cat((state_sample, context_sequenced), dim=2)

        # propagation
        preout, hidden = self.lstm_layer(context_auged_state_sample, hidden)
        out = self.output_layer(preout)
        assert hidden is not None
        return out, hidden
