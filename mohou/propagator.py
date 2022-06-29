from typing import List, Optional

import numpy as np
import torch

from mohou.constant import CONTINUE_FLAG_VALUE
from mohou.encoding_rule import EncodingRule
from mohou.model import LSTM
from mohou.types import ElementDict, TerminateFlag


class Propagator:
    lstm: LSTM
    encoding_rule: EncodingRule
    fed_state_list: List[np.ndarray]  # eatch state is equipped with flag
    n_init_duplicate: int
    is_initialized: bool
    static_context: Optional[np.ndarray] = None
    prop_hidden: bool = False
    _hidden: Optional[torch.Tensor] = None  # hidden state of lstm

    def __init__(
        self,
        lstm: LSTM,
        encoding_rule: EncodingRule,
        n_init_duplicate: int = 0,
        prop_hidden: bool = False,
    ):
        self.lstm = lstm
        self.encoding_rule = encoding_rule
        self.fed_state_list = []
        self.n_init_duplicate = n_init_duplicate
        self.prop_hidden = prop_hidden

        self.is_initialized = False

        require_static_context = lstm.config.n_static_context > 0

        if not require_static_context:  # auto set
            self.static_context = np.empty((0,))

    @property
    def require_static_context(self) -> bool:
        return self.lstm.config.n_static_context > 0

    def set_static_context(self, value: np.ndarray) -> None:
        assert value.ndim == 1
        assert self.lstm.config.n_static_context == len(value)
        self.static_context = value

    def feed(self, elem_dict: ElementDict):
        self._feed(elem_dict)
        if not self.is_initialized:
            for _ in range(self.n_init_duplicate):
                self._feed(elem_dict)

        if not self.is_initialized:
            self.is_initialized = True

    def _feed(self, elem_dict: ElementDict):
        if TerminateFlag not in elem_dict:
            elem_dict[TerminateFlag] = TerminateFlag.from_bool(False)
        state_with_flag = self.encoding_rule.apply(elem_dict)
        self.fed_state_list.append(state_with_flag)

    def predict(self, n_prop: int) -> List[ElementDict]:
        pred_state_list = self._predict(n_prop)
        elem_dict_list = []
        for pred_state in pred_state_list:
            elem_dict = self.encoding_rule.inverse_apply(pred_state)
            elem_dict_list.append(elem_dict)
        return elem_dict_list

    def _predict(self, n_prop: int, force_continue: bool = False) -> List[np.ndarray]:
        pred_state_list: List[np.ndarray] = []

        assert self.static_context is not None, "forgot setting static_context ??"
        context_torch = torch.from_numpy(self.static_context).float().unsqueeze(dim=0)

        for i in range(n_prop):
            states = np.vstack(self.fed_state_list + pred_state_list)
            states_torch = torch.from_numpy(states).float().unsqueeze(dim=0)

            out, hidden = self.lstm.forward(states_torch, context_torch, self._hidden)

            if self.prop_hidden:
                # From the definition, propagating also hidden with the state is supporsed to
                # yield better prediction. But, from my experiment, it seems that performance
                # is actually slight worse than not doing it.
                # But need more experiment.
                self._hidden = hidden

            state_pred_torch = out[0, -1, :]
            state_pred = state_pred_torch.detach().numpy()

            if force_continue:
                state_pred[-1] = CONTINUE_FLAG_VALUE

            pred_state_list.append(state_pred)

        return pred_state_list
