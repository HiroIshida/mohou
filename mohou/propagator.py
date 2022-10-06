from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Generic, List, Optional, Tuple, Type, TypeVar

import numpy as np
import torch

from mohou.constant import CONTINUE_FLAG_VALUE
from mohou.encoder import ImageEncoder
from mohou.encoding_rule import EncodingRule, EncodingRuleBase
from mohou.model import LSTM, PBLSTM
from mohou.model.common import ModelT
from mohou.model.experimental import DisentangleLSTM, ProportionalModel
from mohou.trainer import TrainCache
from mohou.types import ElementDict, RGBImage, TerminateFlag
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
    propagator_model: ModelT
    encoding_rule: EncodingRuleBase

    @classmethod
    @abstractmethod
    def create_default(cls: Type[PropagatorT], porject_path: Path) -> PropagatorT:
        pass

    @property
    @abstractmethod
    def require_static_context(self) -> bool:
        pass

    @abstractmethod
    def feed(self, elem_dict: ElementDict) -> None:
        pass

    @abstractmethod
    def predict(self, n_prop: int) -> List[ElementDict]:
        pass

    @abstractmethod
    def reset(self) -> None:
        pass


class LSTMPropagatorBase(PropagatorBase[ModelT]):
    fed_state_list: List[np.ndarray]  # eatch state is equipped with flag
    n_init_duplicate: int
    is_initialized: bool
    predicted_terminated_flags: List[TerminateFlag]
    static_context: np.ndarray
    prop_hidden: bool = False
    _hidden: Optional[torch.Tensor] = None  # hidden state of lstm

    def __init__(
        self,
        lstm: ModelT,
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

        if not self.require_static_context:  # auto set
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
        if hasattr(self.propagator_model.config, "n_static_context"):
            return self.propagator_model.config.n_static_context > 0
        else:
            return False

    def set_static_context(self, value: np.ndarray) -> None:
        assert hasattr(self.propagator_model.config, "n_static_context")
        assert value.ndim == 1
        assert self.propagator_model.config.n_static_context == len(value)
        self.static_context = value

    def feed(self, elem_dict: ElementDict) -> None:
        self._feed(elem_dict)
        if not self.is_initialized:
            for _ in range(self.n_init_duplicate):
                self._feed(elem_dict)

        if not self.is_initialized:
            self.is_initialized = True

    def _feed(self, elem_dict: ElementDict) -> None:
        if TerminateFlag not in elem_dict:
            elem_dict[TerminateFlag] = TerminateFlag.from_bool(False)
        state_with_flag = self.encoding_rule.apply(elem_dict)
        self.fed_state_list.append(state_with_flag)

    def predict(self, n_prop: int, term_threshold: Optional[float] = None) -> List[ElementDict]:
        # prediction
        pred_state_list = self._predict(n_prop)
        elem_dict_list = []
        for pred_state in pred_state_list:
            elem_dict = self.encoding_rule.inverse_apply(pred_state)
            elem_dict_list.append(elem_dict)

        # update predicted_terminated_flags
        self.predicted_terminated_flags.append(elem_dict_list[0][TerminateFlag])

        return elem_dict_list

    def _predict(
        self, n_prop: int, force_continue: bool = False, term_threshold: Optional[float] = None
    ) -> List[np.ndarray]:
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

            if term_threshold is None:
                satisfy_break_condition = False
            else:
                satisfy_break_condition = state_pred[-1].item() > term_threshold

            if force_continue:
                state_pred[-1] = CONTINUE_FLAG_VALUE
            pred_state_list.append(state_pred)

            if satisfy_break_condition:
                break

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

    @abstractmethod
    def _forward(self, states: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        pass


PropagatorBaseT = TypeVar("PropagatorBaseT", bound=LSTMPropagatorBase)


class _LSTMPropagator(LSTMPropagatorBase[LSTM]):
    def _forward(self, states: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        def numpy_to_unsqueezed_torch(arr: np.ndarray) -> torch.Tensor:
            device = self.get_device()
            return torch.from_numpy(arr).float().unsqueeze(dim=0).to(device)

        context_torch = numpy_to_unsqueezed_torch(self.static_context)
        states_torch = numpy_to_unsqueezed_torch(states)
        out, hidden = self.propagator_model.forward(states_torch, context_torch, self._hidden)
        return out, hidden


class LSTMPropagator(_LSTMPropagator):
    @classmethod
    def create_default(cls, project_path: Path) -> "LSTMPropagator":
        tcach_lstm = TrainCache.load(project_path, LSTM)
        encoding_rule = EncodingRule.create_default(project_path)
        return cls(tcach_lstm.best_model, encoding_rule)


class ChimeraPropagator(_LSTMPropagator):
    @classmethod
    def create_default(cls, project_path: Path) -> "ChimeraPropagator":
        from mohou.model.chimera import Chimera

        tcache_chimera = TrainCache.load(project_path, Chimera)
        chimera_model = tcache_chimera.best_model

        rule = EncodingRule.create_default(project_path)
        rule[RGBImage] = ImageEncoder.from_auto_encoder(chimera_model.ae)
        return cls(chimera_model.lstm, rule)


class PBLSTMPropagator(LSTMPropagatorBase[PBLSTM]):
    parametric_bias: np.ndarray

    @classmethod
    def create_default(cls, project_path: Path) -> "PBLSTMPropagator":
        tcach_lstm = TrainCache.load(project_path, PBLSTM)
        encoding_rule = EncodingRule.create_default(project_path)
        prop = cls(tcach_lstm.best_model, encoding_rule)
        prop.set_pb_to_zero()
        return prop

    def set_parametric_bias(self, value: np.ndarray) -> None:
        assert value.ndim == 1
        assert len(value) == self.propagator_model.config.n_pb_dim
        self.parametric_bias = value

    def set_pb_to_zero(self) -> None:
        n_pb_dim = self.propagator_model.config.n_pb_dim
        vec = np.zeros(n_pb_dim)
        self.set_parametric_bias(vec)

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


class DisentangleLSTMPropagator(LSTMPropagatorBase[DisentangleLSTM]):
    @classmethod
    def create_default(cls, project_path: Path) -> "DisentangleLSTMPropagator":
        tcache = TrainCache.load(project_path, DisentangleLSTM)
        encoding_rule = EncodingRule.create_default(project_path)
        return cls(tcache.best_model, encoding_rule)

    def _forward(self, states: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        def numpy_to_unsqueezed_torch(arr: np.ndarray) -> torch.Tensor:
            device = self.get_device()
            return torch.from_numpy(arr).float().unsqueeze(dim=0).to(device)

        states_torch = numpy_to_unsqueezed_torch(states)
        pred_torch = self.propagator_model.forward(states_torch)
        hidden = torch.empty((0,)).to(self.get_device())  # dummy
        return pred_torch, hidden


@dataclass
class ProportionalModelPropagator(PropagatorBase[ProportionalModel]):
    model: ProportionalModel
    encoding_rule: EncodingRuleBase
    fed_vector: Optional[np.ndarray] = None

    @classmethod
    def create_default(cls, project_path: Path) -> "ProportionalModelPropagator":
        tcache = TrainCache.load(project_path, ProportionalModel)
        encoding_rule = EncodingRule.create_default(project_path)
        return cls(tcache.best_model, encoding_rule)

    @property
    def require_static_context(self) -> bool:
        # currently not support static context
        return False

    def feed(self, elem_dict: ElementDict) -> None:
        feature_vector = self.encoding_rule.apply(elem_dict)
        self.fed_vector = feature_vector

    def predict(self, n_prop: int) -> List[ElementDict]:
        assert self.fed_vector is not None
        feeding_vector_list = [self.fed_vector]
        for _ in range(n_prop):
            latest_vector = feeding_vector_list[-1]
            torch_vector = torch.from_numpy(latest_vector).float()
            seq_sample = torch_vector.unsqueeze(0).unsqueeze(0)
            pred_sample = self.model.forward(seq_sample).squeeze()
            feeding_vector_list.append(pred_sample.cpu().detach().numpy())
        pred_vector_list = feeding_vector_list[1:]
        pred_edict_list = [self.encoding_rule.inverse_apply(vec) for vec in pred_vector_list]
        return pred_edict_list

    def reset(self) -> None:
        pass


class PropagatorSelection(Enum):
    lstm = LSTMPropagator
    pblstm = PBLSTMPropagator
    chimera = ChimeraPropagator
    proportional = ProportionalModelPropagator
    disentangle = DisentangleLSTMPropagator
