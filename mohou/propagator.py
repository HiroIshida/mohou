import numpy as np
import torch
from typing import List

from mohou.constant import CONTINUE_FLAG_VALUE
from mohou.types import ElementDict, TerminateFlag
from mohou.model import LSTM
from mohou.embedding_rule import EmbeddingRule


class Propagator:
    lstm: LSTM
    embed_rule: EmbeddingRule
    fed_state_list: List[np.ndarray]  # eatch state is equipped with flag
    n_init_duplicate: int
    is_initialized: bool

    def __init__(self, lstm: LSTM, embed_rule: EmbeddingRule, n_init_duplicate: int = 0):
        self.lstm = lstm
        self.embed_rule = embed_rule
        self.fed_state_list = []
        self.n_init_duplicate = n_init_duplicate
        self.is_initialized = False

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
        state_with_flag = self.embed_rule.apply(elem_dict)
        self.fed_state_list.append(state_with_flag)

    def predict(self, n_prop: int) -> List[ElementDict]:
        pred_state_list = self._predict(n_prop)
        elem_dict_list = []
        for pred_state in pred_state_list:
            elem_dict = self.embed_rule.inverse_apply(pred_state)
            elem_dict_list.append(elem_dict)
        return elem_dict_list

    def _predict(self, n_prop: int, force_continue: bool = False) -> List[np.ndarray]:
        pred_state_list: List[np.ndarray] = []

        for i in range(n_prop):
            states = np.vstack(self.fed_state_list + pred_state_list)
            states_torch = torch.from_numpy(states).float().unsqueeze(dim=0)
            state_pred_torch: torch.Tensor = self.lstm(states_torch)[:, -1]
            state_pred = state_pred_torch.squeeze().detach().numpy()

            if force_continue:
                state_pred[-1] = CONTINUE_FLAG_VALUE

            pred_state_list.append(state_pred)

        return pred_state_list
