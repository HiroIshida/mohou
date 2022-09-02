import copy
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Type, Union

import numpy as np
import torch
from torch.utils.data import Dataset

from mohou.encoding_rule import EncodingRule
from mohou.types import AngleVector, ElementBase, EpisodeBundle, TerminateFlag
from mohou.utils import AnyT, assert_equal_with_message, assert_seq_list_list_compatible

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
    cov_scale: float

    @classmethod
    def from_seqs(cls, seq_list: List[np.ndarray], cov_scale: float, take_diff: bool = True):
        """construct augmentor.
        seq_list: sequence list used to compute covariance matrix
        take_diff: if True, covaraince is computed using state difference (e.g. {s[t] - s[t-1]}_{1:T})
        otherwise covariance is computed using s[t]_{1:T} sequence.
        """
        if take_diff:
            covmat = cls.compute_diff_covariance(seq_list)
        else:
            covmat = cls.compute_covariance(seq_list)
        return cls(covmat, cov_scale)

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

    def apply(self, seq: np.ndarray) -> np.ndarray:
        """apply augmentation
        seq: 2dim array (n_seqlen, n_dim)
        """
        assert seq.ndim == 2
        assert seq.shape[1] == self.covmat.shape[0]

        covmat_scaled = self.covmat * self.cov_scale**2

        seq_copied = copy.deepcopy(seq)

        n_seqlen, n_dim = seq_copied.shape
        mean = np.zeros(n_dim)
        noises = np.random.multivariate_normal(mean, covmat_scaled, n_seqlen)
        return seq_copied + noises


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


@dataclass
class AutoRegressiveDatasetConfig(SequenceDatasetConfig):
    n_dummy_after_termination: int = 20
    window_size: Optional[int] = None  # if None, all dataset will be just used.
    av_calibration_bias_std: float = 0.0


@dataclass
class AngleVectorCalibrationBiasRandomizer:
    type_bound_table: Dict[Type[ElementBase], slice]
    bias_std: float

    def apply(self, seq: np.ndarray) -> np.ndarray:
        """apply augmentor seq => seq unlike in SequenceDataAugmentor seq => List[seq]"""
        bound = self.type_bound_table[AngleVector]
        av_dim = bound.stop - bound.start
        calibration_error_vec = np.random.randn(av_dim) * self.bias_std

        seq_new = copy.deepcopy(seq)
        seq_new[:, bound] = seq_new[:, bound] + calibration_error_vec
        return seq_new


@dataclass
class AutoRegressiveDataset(Dataset):
    state_seq_list: List[np.ndarray]  # with flag info
    episode_index_list: List[int]
    static_context_list: List[np.ndarray]
    encoding_rule: EncodingRule

    def __post_init__(self):
        assert_equal_with_message(
            len(self.static_context_list), len(self.state_seq_list), "length of sequence"
        )

        # state sequence consists of n_seqlen x n_dim
        assert_equal_with_message(self.state_seq_list[0].ndim, 2, "dimension of state sequence")
        # context sequence consists of n_dim (because its static througout the context list)
        assert_equal_with_message(self.static_context_list[0].ndim, 1, "dimension of context")

    def __len__(self) -> int:
        return len(self.state_seq_list)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """return tuple of episode index of the state_seq, state_esq, and static context"""
        state_seq = torch.from_numpy(self.state_seq_list[idx]).float()
        context = torch.from_numpy(self.static_context_list[idx]).float()
        episode_index = torch.tensor(self.episode_index_list[idx], dtype=torch.int32)
        return episode_index, state_seq, context

    @classmethod
    def from_bundle(
        cls,
        bundle: EpisodeBundle,
        encoding_rule: EncodingRule,
        dataset_config: Optional[AutoRegressiveDatasetConfig] = None,
        static_context_list: Optional[List[np.ndarray]] = None,
    ) -> "AutoRegressiveDataset":

        last_key_of_rule = list(encoding_rule.keys())[-1]
        assert last_key_of_rule == TerminateFlag
        if dataset_config is None:
            dataset_config = AutoRegressiveDatasetConfig()

        state_seq_list = encoding_rule.apply_to_episode_bundle(bundle)

        # setting up static contect
        if static_context_list is None:  # create sequence of 0-dim vector
            static_context_list = [np.zeros((0)) for _ in range(len(state_seq_list))]
        assert_equal_with_message(
            len(static_context_list), len(state_seq_list), "length of sequence"
        )

        # augmentation
        randomizer1 = SequenceDataAugmentor.from_seqs(state_seq_list, dataset_config.cov_scale)
        randomizer2 = AngleVectorCalibrationBiasRandomizer(
            encoding_rule.type_bound_table, dataset_config.av_calibration_bias_std
        )

        episode_index_list_auged = []
        state_seq_list_auged = []
        static_context_list_auged = []

        for episode_index in range(len(state_seq_list)):
            seq = state_seq_list[episode_index]
            static_context = static_context_list[episode_index]

            # append the original
            episode_index_list_auged.append(episode_index)
            state_seq_list_auged.append(seq)
            static_context_list_auged.append(static_context)

            # append the augmented
            for _ in range(dataset_config.n_aug):
                seq_randomized = randomizer2.apply(randomizer1.apply(seq))

                episode_index_list_auged.append(episode_index)
                state_seq_list_auged.append(seq_randomized)
                static_context_list_auged.append(copy.deepcopy(static_context))

        if dataset_config.window_size is None:
            # make all sequence to the same length due to torch batch computation requirement
            aligner = PaddingSequenceAligner.from_seqs(
                state_seq_list_auged, dataset_config.n_dummy_after_termination
            )
            state_seq_list_auged_adjusted = [aligner.apply(seq) for seq in state_seq_list_auged]

            return cls(
                state_seq_list_auged_adjusted,
                episode_index_list_auged,
                static_context_list_auged,
                encoding_rule,
            )
        else:
            # split sequence by window size (Experimental feature!)
            window_state_seq_list = []
            window_episode_index_list = []
            window_static_context_list = []

            window_size = dataset_config.window_size

            for seq_idx in range(len(state_seq_list_auged)):
                state_seq = state_seq_list_auged[seq_idx]
                episode_idx = episode_index_list_auged[seq_idx]
                context = static_context_list_auged[seq_idx]

                n_window = len(state_seq) - window_size + 1
                for window_idx in range(n_window):
                    window_state_seq_list.append(state_seq[window_idx : window_idx + window_size])
                    window_episode_index_list.append(episode_idx)
                    window_static_context_list.append(context)
            return cls(
                window_state_seq_list,
                window_episode_index_list,
                window_static_context_list,
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

        # augmentation
        ctrl_augmentor = SequenceDataAugmentor.from_seqs(
            ctrl_seq_list, config.cov_scale, take_diff=False
        )
        obs_augmentor = SequenceDataAugmentor.from_seqs(
            obs_seq_list, config.cov_scale, take_diff=True
        )

        ctrl_seq_list_auged = []
        obs_seq_list_auged = []
        for ctrl_seq, obs_seq in zip(ctrl_seq_list, obs_seq_list):
            ctrl_seq_list_auged.append(ctrl_seq)
            obs_seq_list_auged.append(obs_seq)
            for _ in range(config.n_aug):
                ctrl_seq_list_auged.append(ctrl_augmentor.apply(ctrl_seq))
                obs_seq_list_auged.append(obs_augmentor.apply(obs_seq))

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
