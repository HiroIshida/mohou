import copy
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, List, Optional, Tuple, Type, Union

import numpy as np
import torch
from torch.utils.data import Dataset

from mohou.encoding_rule import EncodingRule
from mohou.types import ImageT, MultiEpisodeChunk, TerminateFlag
from mohou.utils import assert_two_sequences_same_length

logger = logging.getLogger(__name__)


@dataclass
class AutoEncoderDatasetConfig:
    batch_augment_factor: int = 2  # if you have large enough RAM, set to large (like 4)

    def __post_init__(self):
        assert self.batch_augment_factor >= 0
        logger.info("autoencoder dataset config: {}".format(self))


@dataclass
class AutoEncoderDataset(Dataset, Generic[ImageT]):
    image_type: Type[ImageT]
    image_list: List[ImageT]

    def __len__(self) -> int:
        return len(self.image_list)

    def __getitem__(self, idx) -> torch.Tensor:
        return self.image_list[idx].to_tensor()

    @classmethod
    def from_chunk(
        cls,
        chunk: MultiEpisodeChunk,
        image_type: Type[ImageT],
        augconfig: Optional[AutoEncoderDatasetConfig] = None,
    ) -> "AutoEncoderDataset":

        if augconfig is None:
            augconfig = AutoEncoderDatasetConfig()

        image_list: List[ImageT] = []
        for episode_data in chunk:
            image_list.extend(episode_data.get_sequence_by_type(image_type))

        image_list_rand = copy.deepcopy(image_list)
        for i in range(augconfig.batch_augment_factor):
            image_list_rand.extend([copy.deepcopy(image).randomize() for image in image_list])

        return cls(image_type, image_list_rand)


@dataclass
class AutoRegressiveDatasetConfig:
    n_augmentation: int = 20
    n_dummy_after_termination: int = 20
    cov_scale: float = 0.1

    def __post_init__(self):
        assert self.n_augmentation >= 0
        logger.info("ar dataset config: {}".format(self))


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
class AutoRegressiveDataset(Dataset):
    state_seq_list: List[np.ndarray]  # with flag info
    weight_seq_list: List[np.ndarray]
    encoding_rule: EncodingRule

    def __len__(self) -> int:
        return len(self.state_seq_list)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        state = torch.from_numpy(self.state_seq_list[idx]).float()
        weight = torch.tensor(self.weight_seq_list[idx]).float()
        return state, weight

    @classmethod
    def from_chunk(
        cls,
        chunk: MultiEpisodeChunk,
        encoding_rule: EncodingRule,
        augconfig: Optional[AutoRegressiveDatasetConfig] = None,
        weighting: Optional[Union[WeightPolicy, List[np.ndarray]]] = None,
    ) -> "AutoRegressiveDataset":

        last_key_of_rule = list(encoding_rule.keys())[-1]
        assert last_key_of_rule == TerminateFlag
        if augconfig is None:
            augconfig = AutoRegressiveDatasetConfig()

        state_seq_list = encoding_rule.apply_to_multi_episode_chunk(chunk)

        if weighting is None:
            weighting = ConstantWeightPolicy()

        if isinstance(weighting, list):
            weight_seq_list: List[np.ndarray] = weighting
            logger.info("use user-provided numpy weighting")
        else:
            logger.info("use weight policy: {}".format(weighting))
            weight_seq_list = [weighting(len(seq)) for seq in state_seq_list]

        assert_two_sequences_same_length(state_seq_list, weight_seq_list)

        state_seq_list_auged, weight_seq_list_auged = cls.augment_data(
            state_seq_list, weight_seq_list, augconfig
        )

        state_seq_list_auged_adjusted, weight_seq_list_auged_adjusted = cls.make_same_length(
            state_seq_list_auged, weight_seq_list_auged, augconfig
        )
        return cls(state_seq_list_auged_adjusted, weight_seq_list_auged_adjusted, encoding_rule)

    @staticmethod
    def make_same_length(
        state_seq_list: List[np.ndarray],
        weight_seq_list: List[np.ndarray],
        augconfig: AutoRegressiveDatasetConfig,
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Makes all sequences have the same length"""

        n_max_in_dataset_raw = max([len(seq) for seq in state_seq_list])
        n_max_in_dataset = n_max_in_dataset_raw + augconfig.n_dummy_after_termination

        for i in range(len(state_seq_list)):
            state_seq = state_seq_list[i]
            weight_seq = weight_seq_list[i]

            n_seq = len(state_seq)
            n_padding = n_max_in_dataset - n_seq

            padding_state_seq = np.tile(state_seq[-1], (n_padding, 1))
            padded_state_seq = np.vstack((state_seq, padding_state_seq))

            padding_weight_seq = np.array([weight_seq[-1]] * n_padding)
            padded_weight_seq = np.hstack((weight_seq, padding_weight_seq))
            assert len(padded_state_seq) == n_max_in_dataset
            assert len(padded_weight_seq) == n_max_in_dataset

            state_seq_list[i] = padded_state_seq
            weight_seq_list[i] = padded_weight_seq

        assert_two_sequences_same_length(state_seq_list, weight_seq_list)
        return state_seq_list, weight_seq_list

    @staticmethod
    def trajectory_noise_covariance(state_seq_list: List[np.ndarray]) -> np.ndarray:

        state_diff_list = []
        for state_seq in state_seq_list:
            diff = state_seq[1:, :] - state_seq[:-1, :]
            state_diff_list.append(diff)
        state_diffs = np.vstack(state_diff_list)
        cov_mat = np.cov(state_diffs.T)
        return cov_mat

    @classmethod
    def augment_data(
        cls,
        state_seq_list: List[np.ndarray],
        weight_seq_list: List[np.ndarray],
        augconfig: AutoRegressiveDatasetConfig,
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Augment sequence by adding trajectry noise"""

        cov_mat = cls.trajectory_noise_covariance(state_seq_list)
        cov_mat_scaled = cov_mat * augconfig.cov_scale**2

        noised_state_seq_list = []
        for _ in range(augconfig.n_augmentation):
            for state_seq in state_seq_list:
                n_seq, n_dim = state_seq.shape
                mean = np.zeros(n_dim)
                noise_seq = np.random.multivariate_normal(mean, cov_mat_scaled, n_seq)
                noised_state_seq_list.append(state_seq + noise_seq)

        state_seq_list_auged = copy.deepcopy(state_seq_list)
        state_seq_list_auged.extend(noised_state_seq_list)

        weight_seq_list_auged = copy.deepcopy(weight_seq_list)
        for _ in range(augconfig.n_augmentation):
            for weight_seq in weight_seq_list:
                weight_seq_list_auged.append(copy.deepcopy(weight_seq))

        assert_two_sequences_same_length(state_seq_list_auged, weight_seq_list_auged)

        return state_seq_list_auged, weight_seq_list_auged


@dataclass
class MarkovControlSystemDatasetConfig:
    n_augmentation: int = 20
    cov_scale: float = 0.1

    def __post_init__(self):
        assert self.n_augmentation >= 0
        logger.info("ar dataset config: {}".format(self))


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
        config: Optional[MarkovControlSystemDatasetConfig] = None,
        diff_as_control: bool = True,
    ) -> "MarkovControlSystemDataset":

        ctrl_seq_list = control_encoding_rule.apply_to_multi_episode_chunk(chunk)
        obs_seq_list = observation_encoding_rule.apply_to_multi_episode_chunk(chunk)

        assert_two_sequences_same_length(ctrl_seq_list, obs_seq_list)

        inp_ctrl_seq = []
        inp_obs_seq = []
        out_obs_seq = []
        for i in range(len(ctrl_seq_list)):
            ctrl_seq = ctrl_seq_list[i]
            obs_seq = obs_seq_list[i]
            for j in range(len(ctrl_seq) - 1):
                if diff_as_control:
                    inp_ctrl_seq.append(ctrl_seq[j + 1] - ctrl_seq[j])
                else:
                    inp_ctrl_seq.append(ctrl_seq[j])
                inp_obs_seq.append(obs_seq[j])
                out_obs_seq.append(obs_seq[j + 1])

        return cls(np.array(inp_ctrl_seq), np.array(inp_obs_seq), np.array(out_obs_seq))
