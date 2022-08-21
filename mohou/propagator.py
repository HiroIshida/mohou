from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, List, Optional, Tuple, Type, TypeVar

import numpy as np
import torch

from mohou.constant import CONTINUE_FLAG_VALUE
from mohou.encoding_rule import EncodingRule
from mohou.model import LSTM, PBLSTM
from mohou.model.lstm import LSTMBaseT
from mohou.types import ElementDict, TerminateFlag


@dataclass
class TerminateChecker:
    n_check_flag: int = 3
    terminate_threshold: float = 0.95

    def __call__(self, flags_predicted: List[TerminateFlag]) -> bool:
        # checking only latest terminate flags is not sufficient because
        # sometimes terminate flag value experiences sudden surge to 1 and
        # immediently goes back 0.
        if len(flags_predicted) < self.n_check_flag:
            return False

        flag_arr = np.array([flag.numpy().item() for flag in flags_predicted])
        return np.all(flag_arr[-self.n_check_flag :] > self.terminate_threshold).item()


class PropagatorBase(ABC, Generic[LSTMBaseT]):
    lstm_model: LSTMBaseT
    encoding_rule: EncodingRule
    fed_state_list: List[np.ndarray]  # eatch state is equipped with flag
    n_init_duplicate: int
    is_initialized: bool
    predicted_terminated_flags: List[TerminateFlag]
    static_context: Optional[np.ndarray] = None
    prop_hidden: bool = False
    _hidden: Optional[torch.Tensor] = None  # hidden state of lstm

    def __init__(
        self,
        lstm: LSTMBaseT,
        encoding_rule: EncodingRule,
        n_init_duplicate: int = 0,
        prop_hidden: bool = False,
    ):
        self.lstm_model = lstm
        self.encoding_rule = encoding_rule
        self.fed_state_list = []
        self.n_init_duplicate = n_init_duplicate
        self.prop_hidden = prop_hidden

        self.is_initialized = False
        self.predicted_terminated_flags = []

        require_static_context = lstm.config.n_static_context > 0

        if not require_static_context:  # auto set
            self.static_context = np.empty((0,))

    def is_terminated(self) -> bool:
        checker = TerminateChecker()
        return checker(self.predicted_terminated_flags)

    def clear_fed_data(self) -> None:
        self.fed_state_list = []
        self.is_initialized = False

    @property
    def require_static_context(self) -> bool:
        return self.lstm_model.config.n_static_context > 0

    def set_static_context(self, value: np.ndarray) -> None:
        assert value.ndim == 1
        assert self.lstm_model.config.n_static_context == len(value)
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
        # prediction
        pred_state_list = self._predict(n_prop)
        elem_dict_list = []
        for pred_state in pred_state_list:
            elem_dict = self.encoding_rule.inverse_apply(pred_state)
            elem_dict_list.append(elem_dict)

        # update predicted_terminated_flags
        self.predicted_terminated_flags.append(elem_dict_list[0][TerminateFlag])

        return elem_dict_list

    def _predict(self, n_prop: int, force_continue: bool = False) -> List[np.ndarray]:
        pred_state_list: List[np.ndarray] = []

        assert self.static_context is not None, "forgot setting static_context ??"

        for i in range(n_prop):
            states = np.vstack(self.fed_state_list + pred_state_list)
            out, hidden = self._forward(states)

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

    @classmethod
    @abstractmethod
    def lstm_type(cls) -> Type[LSTMBaseT]:
        pass

    @abstractmethod
    def _forward(self, state: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        pass


PropagatorBaseT = TypeVar("PropagatorBaseT", bound=PropagatorBase)


class Propagator(PropagatorBase[LSTM]):
    @classmethod
    def lstm_type(cls) -> Type[LSTM]:
        return LSTM

    def _forward(self, states: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        context_torch = torch.from_numpy(self.static_context).float().unsqueeze(dim=0)
        states_torch = torch.from_numpy(states).float().unsqueeze(dim=0)
        out, hidden = self.lstm_model.forward(states_torch, context_torch, self._hidden)
        return out, hidden


class PBLSTMPropagator(PropagatorBase[PBLSTM]):
    parametric_bias: np.ndarray

    def set_parametric_bias(self, value: np.ndarray) -> None:
        assert value.ndim == 1
        assert len(value) == self.lstm_model.config.n_pb_dim
        self.parametric_bias = value

    def set_pb_to_zero(self) -> None:
        n_pb_dim = self.lstm_model.config.n_pb_dim
        vec = np.zeros(n_pb_dim)
        self.set_parametric_bias(vec)

    @classmethod
    def lstm_type(cls) -> Type[PBLSTM]:
        return PBLSTM

    def _forward(self, states: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        context_torch = torch.from_numpy(self.static_context).float().unsqueeze(dim=0)
        pb_torch = torch.from_numpy(self.parametric_bias).float().unsqueeze(dim=0)
        states_torch = torch.from_numpy(states).float().unsqueeze(dim=0)
        out, hidden = self.lstm_model.forward(states_torch, pb_torch, context_torch, self._hidden)
        return out, hidden
