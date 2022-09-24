from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, List, Optional, Tuple, Type, TypeVar

import numpy as np
import torch

from mohou.constant import CONTINUE_FLAG_VALUE
from mohou.encoding_rule import EncodingRuleBase
from mohou.model import LSTM, PBLSTM
from mohou.model.common import ModelT
from mohou.model.lstm import LSTMBaseT
from mohou.types import ElementDict, TerminateFlag
from mohou.utils import detect_device


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
        return bool(np.all(flag_arr[-self.n_check_flag :] > self.terminate_threshold).item())


PropagatorT = TypeVar("PropagatorT", bound="PropagatorBase")


class PropagatorBase(ABC, Generic[ModelT]):
    pass


class LSTMPropagatorBase(PropagatorBase[LSTMBaseT]):
    propagator_model: LSTMBaseT
    encoding_rule: EncodingRuleBase
    fed_state_list: List[np.ndarray]  # eatch state is equipped with flag
    n_init_duplicate: int
    is_initialized: bool
    predicted_terminated_flags: List[TerminateFlag]
    static_context: np.ndarray
    prop_hidden: bool = False
    _hidden: Optional[torch.Tensor] = None  # hidden state of lstm

    def __init__(
        self,
        lstm: LSTMBaseT,
        encoding_rule: EncodingRuleBase,
        n_init_duplicate: int = 0,
        prop_hidden: bool = False,
        device: Optional[torch.device] = None,
    ):

        if device is None:
            device = detect_device()
        encoding_rule.set_device(device)
        lstm.put_on_device(device)

        self.propagator_model = lstm
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

    def reset(self) -> None:
        self.fed_state_list = []
        self.is_initialized = False
        self.predicted_terminated_flags = []

    @property
    def require_static_context(self) -> bool:
        return self.propagator_model.config.n_static_context > 0

    def set_static_context(self, value: np.ndarray) -> None:
        assert value.ndim == 1
        assert self.propagator_model.config.n_static_context == len(value)
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
            state_pred = state_pred_torch.cpu().detach().numpy()

            if force_continue:
                state_pred[-1] = CONTINUE_FLAG_VALUE

            pred_state_list.append(state_pred)

        return pred_state_list

    def get_device(self) -> Optional[torch.device]:
        rule_device = self.encoding_rule.get_device()
        lstm_device = self.propagator_model.device

        if rule_device is None:
            return lstm_device
        else:
            assert rule_device == lstm_device
            return rule_device

    def set_device(self, device: torch.device) -> None:
        self.encoding_rule.set_device(device)
        self.propagator_model.put_on_device(device)

    @classmethod
    @abstractmethod
    def lstm_type(cls) -> Type[LSTMBaseT]:
        pass

    @abstractmethod
    def _forward(self, state: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        pass


PropagatorBaseT = TypeVar("PropagatorBaseT", bound=LSTMPropagatorBase)


class Propagator(LSTMPropagatorBase[LSTM]):
    @classmethod
    def lstm_type(cls) -> Type[LSTM]:
        return LSTM

    def _forward(self, states: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        def numpy_to_unsqueezed_torch(arr: np.ndarray) -> torch.Tensor:
            device = self.get_device()
            return torch.from_numpy(arr).float().unsqueeze(dim=0).to(device)

        context_torch = numpy_to_unsqueezed_torch(self.static_context)
        states_torch = numpy_to_unsqueezed_torch(states)
        out, hidden = self.propagator_model.forward(states_torch, context_torch, self._hidden)
        return out, hidden


class PBLSTMPropagator(LSTMPropagatorBase[PBLSTM]):
    parametric_bias: np.ndarray

    def set_parametric_bias(self, value: np.ndarray) -> None:
        assert value.ndim == 1
        assert len(value) == self.propagator_model.config.n_pb_dim
        self.parametric_bias = value

    def set_pb_to_zero(self) -> None:
        n_pb_dim = self.propagator_model.config.n_pb_dim
        vec = np.zeros(n_pb_dim)
        self.set_parametric_bias(vec)

    @classmethod
    def lstm_type(cls) -> Type[PBLSTM]:
        return PBLSTM

    def _forward(self, states: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        def numpy_to_unsqueezed_torch(arr: np.ndarray) -> torch.Tensor:
            device = self.get_device()
            return torch.from_numpy(arr).float().unsqueeze(dim=0).to(device)

        context_torch = numpy_to_unsqueezed_torch(self.static_context)
        pb_torch = numpy_to_unsqueezed_torch(self.parametric_bias)
        states_torch = numpy_to_unsqueezed_torch(states)
        out, hidden = self.propagator_model.forward(
            states_torch, pb_torch, context_torch, self._hidden
        )
        return out, hidden
