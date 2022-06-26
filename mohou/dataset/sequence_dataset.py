import copy
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset

from mohou.encoding_rule import EncodingRule
from mohou.types import MultiEpisodeChunk, TerminateFlag
from mohou.utils import assert_equal_with_message, assert_seq_list_list_compatible

logger = logging.getLogger(__name__)


@dataclass
class SequenceDatasetConfig:
    n_aug: int = 20
    cov_scale: float = 0.1

    def __new__(cls, *args, **kwargs):
        assert len(args) == 0, "please instantiate config only using kwargs"
        return super(SequenceDatasetConfig, cls).__new__(cls)

    def __post_init__(self):
        assert self.n_aug >= 0
        assert self.cov_scale < 1.0
        logger.info("sequence dataset config: {}".format(self))


@dataclass
class SequenceDataAugmentor:  # functor
    config: SequenceDatasetConfig
    take_diff: bool = True

    @staticmethod
    def compute_covariance(state_seq_list: List[np.ndarray]) -> np.ndarray:
        state_diffs = np.vstack(state_seq_list)
        cov_mat = np.cov(state_diffs.T)
        return cov_mat

    @staticmethod
    def compute_diff_covariance(state_seq_list: List[np.ndarray]) -> np.ndarray:

        state_diff_list = []
        for state_seq in state_seq_list:
            diff = state_seq[1:, :] - state_seq[:-1, :]
            state_diff_list.append(diff)
        state_diffs = np.vstack(state_diff_list)
        cov_mat = np.cov(state_diffs.T)
        return cov_mat

    def apply(
        self, state_seq_list: List[np.ndarray], other_seq_list_list: List[List[np.ndarray]]
    ) -> Tuple[List[np.ndarray], List[List[np.ndarray]]]:
        """apply augmentation
        state_seq_list will be randomized.
        each seq_list in other_seq_list_list will not be randomized. But just augmented so that they are
        compatible with augmented state_seq_list.
        """

        if self.take_diff:
            cov_mat = self.compute_diff_covariance(state_seq_list)
        else:
            cov_mat = self.compute_covariance(state_seq_list)
        cov_mat_scaled = cov_mat * self.config.cov_scale**2

        noised_state_seq_list = []
        for _ in range(self.config.n_aug):
            for state_seq in state_seq_list:
                n_seq, n_dim = state_seq.shape
                mean = np.zeros(n_dim)
                noise_seq = np.random.multivariate_normal(mean, cov_mat_scaled, n_seq)
                noised_state_seq_list.append(state_seq + noise_seq)

        state_seq_list_auged = copy.deepcopy(state_seq_list)
        state_seq_list_auged.extend(noised_state_seq_list)

        # Just increase the number keeping its order matches with state_seq_list_auged
        other_seq_list_auged_list = []
        for other_seq_list in other_seq_list_list:
            other_seq_list_auged = copy.deepcopy(other_seq_list)
            for _ in range(self.config.n_aug):
                for other_seq in other_seq_list:
                    other_seq_list_auged.append(copy.deepcopy(other_seq))
            other_seq_list_auged_list.append(other_seq_list_auged)

        return state_seq_list_auged, other_seq_list_auged_list


def make_same_length(
    seq_llist: List[List[np.ndarray]], n_dummy_after_termination: int
) -> List[List[np.ndarray]]:
    """Makes all sequences have the same length"""
    # here llist means list of list
    # check if all sequence list has same length
    seq_llist = copy.deepcopy(seq_llist)
    seqlen_list_reference = [len(seq) for seq in seq_llist[0]]  # first seq_list as ref
    for seq_list in seq_llist:
        seqlen_list = [len(seq) for seq in seq_list]
        assert seqlen_list == seqlen_list_reference

    def _make_same_length(seq_list: List[np.ndarray], n_seqlen_target: int):
        padded_seq_list = []
        for seq in seq_list:
            n_seqlen = len(seq)
            n_padding = n_seqlen_target - n_seqlen
            assert n_padding >= 0

            if n_padding == 0:
                padded_seq = copy.copy(seq)
            else:
                elem_last = seq[-1]
                padding_seq_shape = [n_padding] + [1 for _ in range(elem_last.ndim)]
                padding_seq = np.tile(elem_last, padding_seq_shape)
                padded_seq = np.concatenate((seq, padding_seq), axis=0)

            assert len(padded_seq) == n_seqlen_target
            padded_seq_list.append(padded_seq)
        return padded_seq_list

    n_seqlen_target = max(seqlen_list_reference) + n_dummy_after_termination
    return [_make_same_length(seq_list, n_seqlen_target) for seq_list in seq_llist]


class WeightPolicy(ABC):
    @abstractmethod
    def __call__(self, n_seq_lne: int) -> np.ndarray:
        pass


class ConstantWeightPolicy(WeightPolicy):
    def __call__(self, n_seq_lne: int) -> np.ndarray:
        return np.ones(n_seq_lne)


@dataclass
class PWLinearWeightPolicy(WeightPolicy):
    w_left: float
    w_right: float

    def __call__(self, n_seq_len: int) -> np.ndarray:
        return np.linspace(self.w_left, self.w_right, n_seq_len)


@dataclass
class AutoRegressiveDatasetConfig(SequenceDatasetConfig):
    n_dummy_after_termination: int = 20


@dataclass
class AutoRegressiveDataset(Dataset):
    state_seq_list: List[np.ndarray]  # with flag info
    static_context_list: List[np.ndarray]
    weight_seq_list: List[np.ndarray]
    encoding_rule: EncodingRule

    def __len__(self) -> int:
        return len(self.state_seq_list)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        state = torch.from_numpy(self.state_seq_list[idx]).float()
        context = torch.from_numpy(self.static_context_list[idx]).float()
        weight = torch.tensor(self.weight_seq_list[idx]).float()
        return state, context, weight

    def __post_init__(self):  # validation
        assert_seq_list_list_compatible([self.state_seq_list, self.weight_seq_list])
        assert_equal_with_message(
            len(self.static_context_list), len(self.state_seq_list), "length of sequence"
        )

    @classmethod
    def from_chunk(
        cls,
        chunk: MultiEpisodeChunk,
        encoding_rule: EncodingRule,
        augconfig: Optional[AutoRegressiveDatasetConfig] = None,
        static_context_list: Optional[List[np.ndarray]] = None,
        weighting: Optional[Union[WeightPolicy, List[np.ndarray]]] = None,
    ) -> "AutoRegressiveDataset":

        last_key_of_rule = list(encoding_rule.keys())[-1]
        assert last_key_of_rule == TerminateFlag
        if augconfig is None:
            augconfig = AutoRegressiveDatasetConfig()

        state_seq_list = encoding_rule.apply_to_multi_episode_chunk(chunk)

        # setting up weighting
        if weighting is None:
            weighting = ConstantWeightPolicy()
        if isinstance(weighting, list):
            weight_seq_list: List[np.ndarray] = weighting
            logger.info("use user-provided numpy weighting")
        else:
            logger.info("use weight policy: {}".format(weighting))
            weight_seq_list = [weighting(len(seq)) for seq in state_seq_list]
            assert_seq_list_list_compatible([state_seq_list, weight_seq_list])

        # setting up biases
        if static_context_list is None:  # create sequence of 0-dim vector
            static_context_list = [np.zeros((0)) for _ in range(len(state_seq_list))]
        assert_equal_with_message(
            len(static_context_list), len(state_seq_list), "length of sequence"
        )

        # augmentation
        augmentor = SequenceDataAugmentor(augconfig, take_diff=True)
        state_seq_list_auged, [weight_seq_list_auged, static_context_list_auged] = augmentor.apply(
            state_seq_list, [weight_seq_list, static_context_list]
        )
        assert weight_seq_list_auged is not None  # for mypy

        # make all sequence to the same length due to torch batch computation requirement
        state_seq_list_auged_adjusted, weight_seq_list_auged_adjusted = make_same_length(
            [state_seq_list_auged, weight_seq_list_auged], augconfig.n_dummy_after_termination
        )
        return cls(
            state_seq_list_auged_adjusted,
            static_context_list_auged,
            weight_seq_list_auged_adjusted,
            encoding_rule,
        )


@dataclass
class MarkovControlSystemDataset(Dataset):
    """o_{t+1} = f(o_{t}, u_t{t})"""

    inp_ctrl_seq: np.ndarray
    inp_obs_seq: np.ndarray
    out_obs_seq: np.ndarray

    def __len__(self) -> int:
        return len(self.inp_ctrl_seq)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        inp_ctrl = torch.from_numpy(self.inp_ctrl_seq[idx]).float()
        inp_obs = torch.from_numpy(self.inp_obs_seq[idx]).float()
        out_obs = torch.from_numpy(self.out_obs_seq[idx]).float()
        return inp_ctrl, inp_obs, out_obs

    @classmethod
    def from_chunk(
        cls,
        chunk: MultiEpisodeChunk,
        control_encoding_rule: EncodingRule,
        observation_encoding_rule: EncodingRule,
        config: Optional[SequenceDatasetConfig] = None,
        diff_as_control: bool = True,
    ) -> "MarkovControlSystemDataset":
        if config is None:
            config = SequenceDatasetConfig()

        ctrl_seq_list = control_encoding_rule.apply_to_multi_episode_chunk(chunk)
        obs_seq_list = observation_encoding_rule.apply_to_multi_episode_chunk(chunk)
        assert_seq_list_list_compatible([ctrl_seq_list, obs_seq_list])

        ctrl_augmentor = SequenceDataAugmentor(config, take_diff=False)
        obs_augmentor = SequenceDataAugmentor(config, take_diff=True)

        ctrl_seq_list_auged, _ = ctrl_augmentor.apply(ctrl_seq_list, [])
        obs_seq_list_auged, _ = obs_augmentor.apply(obs_seq_list, [])

        inp_ctrl_seq = []
        inp_obs_seq = []
        out_obs_seq = []
        for i in range(len(ctrl_seq_list_auged)):
            ctrl_seq = ctrl_seq_list_auged[i]
            obs_seq = obs_seq_list_auged[i]
            for j in range(len(ctrl_seq) - 1):
                if diff_as_control:
                    inp_ctrl_seq.append(ctrl_seq[j + 1] - ctrl_seq[j])
                else:
                    inp_ctrl_seq.append(ctrl_seq[j])
                inp_obs_seq.append(obs_seq[j])
                out_obs_seq.append(obs_seq[j + 1])

        return cls(np.array(inp_ctrl_seq), np.array(inp_obs_seq), np.array(out_obs_seq))
