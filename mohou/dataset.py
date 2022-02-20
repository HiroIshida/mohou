from abc import abstractmethod
from dataclasses import dataclass
import logging
from typing import List, Generic, Type, TypeVar, Optional

import albumentations as al
import numpy as np
import torch
from torch.utils.data import Dataset

from mohou.constant import CONTINUE_FLAG_VALUE, END_FLAG_VALUE
from mohou.embedding_rule import EmbeddingRule
from mohou.types import ElementSequence, ImageT, MultiEpisodeChunk, RGBImage

logger = logging.getLogger(__name__)

AutoEncoderDataestT = TypeVar('AutoEncoderDataestT', bound='AutoEncoderDataset')


class MohouDataset(Dataset):

    @abstractmethod
    def update_dataset(self) -> None:
        pass


class AutoEncoderDataset(MohouDataset, Generic[ImageT]):
    image_type: Type[ImageT]
    image_list: List[ImageT]
    image_list_rand: List[ImageT]

    def __init__(self, image_list: List[ImageT]):
        self.image_list = image_list
        self.update_dataset()

    def __len__(self) -> int:
        return len(self.image_list)

    def __getitem__(self, idx) -> torch.Tensor:
        return self.image_list[idx].to_tensor()

    def update_dataset(self):
        logger.info('randomizing data...')
        self.image_list_rand = self._randomize_data(self.image_list)

    @classmethod
    def from_chunk(
            cls: Type[AutoEncoderDataestT],
            chunk: MultiEpisodeChunk) -> AutoEncoderDataestT:

        image_list: List[ImageT] = []
        for episode_data in chunk:
            image_seq: ElementSequence[ImageT] = episode_data.filter_by_type(cls.image_type)
            image_list.extend(image_seq)

        return cls(image_list)

    @staticmethod
    @abstractmethod
    def _randomize_data(image_list: List[ImageT]) -> List[ImageT]:
        pass


class RGBAutoEncoderDataset(AutoEncoderDataset[RGBImage]):
    image_type = RGBImage

    @staticmethod
    def _randomize_data(image_list: List[ImageT]) -> List[ImageT]:
        # TODO(HiroIshida) make them config
        aug_guass = al.GaussNoise(p=1)
        aug_rgbshit = al.RGBShift(r_shift_limit=40, g_shift_limit=40, b_shift_limit=40)
        aug_composed = al.Compose([aug_guass, aug_rgbshit])

        randomized_image_list = [aug_composed(image=img)['image'] for img in image_list]
        return randomized_image_list


@dataclass
class AutoRegressiveAugConfig:
    n_augmentation: int = 20
    cov_scale: float = 0.1


class AutoRegressiveDataset(MohouDataset):
    state_seq_list: List[np.ndarray]

    def __init__(self, state_seq_list: List[np.ndarray]):
        self.state_seq_list = self.attach_flag_info(state_seq_list)

    def __len__(self) -> int:
        return len(self.state_seq_list)

    def __getitem__(self, idx) -> torch.Tensor:
        return torch.from_numpy(self.state_seq_list[idx]).float()

    def update_dataset(self) -> None:
        pass

    @classmethod
    def from_chunk(
            cls,
            chunk: MultiEpisodeChunk,
            embed_rule: EmbeddingRule,
            augconfig: Optional[AutoRegressiveAugConfig] = None) -> 'AutoRegressiveDataset':

        if augconfig is None:
            augconfig = AutoRegressiveAugConfig()

        state_seq_list = embed_rule.apply_to_multi_episode_chunk(chunk)
        state_auged_seq_list = cls.augment_data(state_seq_list, augconfig)
        return cls(state_auged_seq_list)

    @staticmethod
    def attach_flag_info(state_seq_list: List[np.ndarray]) -> List[np.ndarray]:
        """Makes all sequences have the same length"""

        n_max_in_dataset = max([len(seq) for seq in state_seq_list])

        for i in range(len(state_seq_list)):
            state_seq = state_seq_list[i]

            n_seq = len(state_seq)
            n_padding = n_max_in_dataset - n_seq

            flag_seq = np.ones((n_max_in_dataset, 1))
            flag_seq[:n_seq] *= CONTINUE_FLAG_VALUE
            flag_seq[n_seq:] *= END_FLAG_VALUE

            padding_state_seq = np.tile(state_seq[-1], (n_padding, 1))
            padded_state_seq = np.vstack((state_seq, padding_state_seq))
            padded_state_flag_seq = np.hstack((padded_state_seq, flag_seq))

            state_seq_list[i] = padded_state_flag_seq

        return state_seq_list

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
            augconfig: AutoRegressiveAugConfig) -> List[np.ndarray]:
        """Augment sequence by adding trajectry noise"""

        cov_mat = cls.trajectory_noise_covariance(state_seq_list)
        cov_mat_scaled = cov_mat * augconfig.cov_scale ** 2

        noised_state_seq_list = []
        for state_seq in state_seq_list:
            n_seq, n_dim = state_seq.shape
            mean = np.zeros(n_dim)
            noise_seq = np.random.multivariate_normal(mean, cov_mat_scaled, n_seq)
            noised_state_seq_list.append(state_seq + noise_seq)

        state_seq_list.extend(noised_state_seq_list)
        return state_seq_list
