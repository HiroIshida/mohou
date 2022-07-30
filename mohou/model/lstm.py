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


class LSTMBaseMixIn:
    """provide config independent fetures"""

    @staticmethod
    def _setup_inner(
        n_input: int, n_output: int, n_hidden: int, n_layer: int, n_output_layer: int
    ) -> Tuple[nn.LSTM, nn.Sequential]:

        lstm_layer = nn.LSTM(n_input, n_hidden, n_layer, batch_first=True)
        output_layers = []
        for _ in range(n_output_layer):
            output_layers.append(nn.Linear(n_hidden, n_hidden))
        output_layers.append(nn.Linear(n_hidden, n_output))
        output_layer = nn.Sequential(*output_layers)
        return lstm_layer, output_layer

    @staticmethod
    def _loss_inner_type_wise(
        state_output_ref: torch.Tensor,
        state_output_pred: torch.Tensor,
        type_bound_table: Dict[Type[ElementBase], slice],
    ) -> LossDict:
        # NOTE: This is an experimental feature. Compute type-wise prediction loss. LossDict looks like
        # d["AngleVector"] = 0.002, d["RGBImage"] = 0.0003
        d = {}
        for elem_type, bound in type_bound_table.items():
            pred_typewise = state_output_pred[:, :, bound]
            state_sample_output_typewise = state_output_ref[:, :, bound]
            loss_value_partial = nn.MSELoss()(pred_typewise, state_sample_output_typewise)
            key = elem_type.__name__
            d[key] = loss_value_partial
        return LossDict(d)

    @staticmethod
    def _loss_inner(state_output_ref: torch.Tensor, state_output_pred: torch.Tensor) -> LossDict:
        loss_value = nn.MSELoss()(state_output_pred, state_output_ref)
        return LossDict({"prediction": loss_value})


class LSTM(LSTMBaseMixIn, ModelBase[LSTMConfig]):
    """
    lstm: x_t+1 = f(x_t, x_t-1, ...)
    lstm with context: x_t+1 = f(x_t, x_t-1, ..., c) where c is static (time-invariant) context
    """

    lstm_layer: nn.LSTM
    output_layer: nn.Sequential

    def _setup_from_config(self, config: LSTMConfig) -> None:
        n_input = config.n_state_with_flag + config.n_static_context
        n_output = config.n_state_with_flag
        self.lstm_layer, self.output_layer = self._setup_inner(
            n_input, n_output, config.n_hidden, config.n_layer, config.n_output_layer
        )

    def loss(self, sample: Tuple[torch.Tensor, torch.Tensor]) -> LossDict:

        state_sample, static_context_sample = sample

        # sanity check
        n_batch, n_seq_len, n_dim = state_sample.shape
        assert static_context_sample.ndim == 2
        assert static_context_sample.shape[0] == n_batch
        assert static_context_sample.shape[1] == self.config.n_static_context

        # propagation
        state_sample_input, state_sample_output = state_sample[:, :-1], state_sample[:, 1:]
        pred, _ = self.forward(state_sample_input, static_context_sample)

        if self.config.type_wise_loss:
            assert self.config.type_bound_table is not None
            return self._loss_inner_type_wise(
                state_sample_output, pred, self.config.type_bound_table
            )
        else:
            return self._loss_inner(state_sample_output, pred)

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
