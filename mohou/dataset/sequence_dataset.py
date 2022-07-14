import copy
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset

from mohou.encoding_rule import EncodingRule
from mohou.types import EpisodeBundle, TerminateFlag
from mohou.utils import (
    AnyT,
    assert_equal_with_message,
    assert_seq_list_list_compatible,
    flatten_lists,
)

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
class SequenceDataAugmentor:
    covmat: np.ndarray
    config: SequenceDatasetConfig

    @classmethod
    def from_seqs(
        cls, seq_list: List[np.ndarray], config: SequenceDatasetConfig, take_diff: bool = True
    ):
        """construct augmentor.
        seq_list: sequence list used to compute covariance matrix
        take_diff: if True, covaraince is computed using state difference (e.g. {s[t] - s[t-1]}_{1:T})
        otherwise covariance is computed using s[t]_{1:T} sequence.
        """
        if take_diff:
            covmat = cls.compute_diff_covariance(seq_list)
        else:
            covmat = cls.compute_covariance(seq_list)
        return cls(covmat, config)

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

    def apply(self, seq: np.ndarray) -> List[np.ndarray]:
        """apply augmentation
        seq: 2dim array (n_seqlen, n_dim)
        """
        assert seq.ndim == 2
        assert seq.shape[1] == self.covmat.shape[0]

        covmat_scaled = self.covmat * self.config.cov_scale**2

        seq_original = copy.deepcopy(seq)
        auged_seq_list: List[np.ndarray] = [seq_original]
        for _ in range(self.config.n_aug):
            n_seqlen, n_dim = seq_original.shape
            mean = np.zeros(n_dim)
            noises = np.random.multivariate_normal(mean, covmat_scaled, n_seqlen)
            noised_seq = seq_original + noises
            auged_seq_list.append(noised_seq)

        return auged_seq_list


@dataclass
class PaddingSequenceAligner:
    n_seqlen_target: int

    @classmethod
    def from_seqs(
        cls, seqs: Union[List[np.ndarray], List[List]], n_after_termination: int
    ) -> "PaddingSequenceAligner":
        n_seqlen_max = max([len(seq) for seq in seqs])
        n_seqlen_target = n_seqlen_max + n_after_termination
        return cls(n_seqlen_target)

    def apply(self, seq: AnyT) -> AnyT:  # TODO: specify type
        """get padded sequence based on seq.
        seq: np.ndarray or List
        if seq is [1, 2, 3, 4] and n_seqlen_target = 10, the output padded sequence looks like
        [1, 2, 3, 4, 4, 4, 4, 4, 4, 4]
        """
        seq = copy.deepcopy(seq)
        n_seqlen = len(seq)
        n_padding = self.n_seqlen_target - n_seqlen
        assert n_padding >= 0

        # TODO(HiroIshida) To remove type-ignores it's better to use singledispatch .
        # however we prefere simpilicy in this case.
        if isinstance(seq, list):
            padded_seq = seq + [copy.deepcopy(seq[-1]) for _ in range(n_padding)]
        elif isinstance(seq, np.ndarray):
            if n_padding == 0:
                padded_seq = seq  # type: ignore
            else:
                elem_last = seq[-1]
                padding_seq_shape = [n_padding] + [1 for _ in range(elem_last.ndim)]
                padding_seq = np.tile(elem_last, padding_seq_shape)
                padded_seq = np.concatenate((seq, padding_seq), axis=0)
        else:
            assert False

        assert len(padded_seq) == self.n_seqlen_target
        return padded_seq  # type: ignore


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
    def from_bundle(
        cls,
        bundle: EpisodeBundle,
        encoding_rule: EncodingRule,
        augconfig: Optional[AutoRegressiveDatasetConfig] = None,
        static_context_list: Optional[List[np.ndarray]] = None,
        weighting: Optional[Union[WeightPolicy, List[np.ndarray]]] = None,
    ) -> "AutoRegressiveDataset":

        last_key_of_rule = list(encoding_rule.keys())[-1]
        assert last_key_of_rule == TerminateFlag
        if augconfig is None:
            augconfig = AutoRegressiveDatasetConfig()

        state_seq_list = encoding_rule.apply_to_episode_bundle(bundle)

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
        augmentor = SequenceDataAugmentor.from_seqs(state_seq_list, augconfig)
        state_seq_list_auged = flatten_lists([augmentor.apply(seq) for seq in state_seq_list])
        weight_seq_list_auged = flatten_lists(
            [[copy.deepcopy(seq) for _ in range(augconfig.n_aug + 1)] for seq in weight_seq_list]
        )  # +1 for original data
        static_context_list_auged = flatten_lists(
            [[copy.deepcopy(c) for _ in range(augconfig.n_aug + 1)] for c in static_context_list]
        )  # +1 for original data

        # make all sequence to the same length due to torch batch computation requirement
        assert_seq_list_list_compatible([state_seq_list_auged, weight_seq_list_auged])
        aligner = PaddingSequenceAligner.from_seqs(
            state_seq_list_auged, augconfig.n_dummy_after_termination
        )
        state_seq_list_auged_adjusted = [aligner.apply(seq) for seq in state_seq_list_auged]
        weight_seq_list_auged_adjusted = [aligner.apply(seq) for seq in weight_seq_list_auged]

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
    def from_bundle(
        cls,
        bundle: EpisodeBundle,
        control_encoding_rule: EncodingRule,
        observation_encoding_rule: EncodingRule,
        config: Optional[SequenceDatasetConfig] = None,
        diff_as_control: bool = True,
    ) -> "MarkovControlSystemDataset":
        if config is None:
            config = SequenceDatasetConfig()

        ctrl_seq_list = control_encoding_rule.apply_to_episode_bundle(bundle)
        obs_seq_list = observation_encoding_rule.apply_to_episode_bundle(bundle)
        assert_seq_list_list_compatible([ctrl_seq_list, obs_seq_list])

        ctrl_augmentor = SequenceDataAugmentor.from_seqs(ctrl_seq_list, config, take_diff=False)
        obs_augmentor = SequenceDataAugmentor.from_seqs(obs_seq_list, config, take_diff=True)

        ctrl_seq_list_auged = flatten_lists([ctrl_augmentor.apply(seq) for seq in ctrl_seq_list])
        obs_seq_list_auged = flatten_lists([obs_augmentor.apply(seq) for seq in obs_seq_list])

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
