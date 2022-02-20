from abc import abstractmethod
from typing import List, Generic, Type, TypeVar

import albumentations as al
import numpy as np
import torch
from torch.utils.data import Dataset

from mohou.constant import CONTINUE_FLAG_VALUE, END_FLAG_VALUE
from mohou.embedding_rule import EmbeddingRule
from mohou.types import ElementSequence, ImageT, MultiEpisodeChunk, RGBImage

AutoEncoderDataestT = TypeVar('AutoEncoderDataestT', bound='AutoEncoderDataset')


class AutoEncoderDataset(Dataset, Generic[ImageT]):
    image_type: Type[ImageT]
    image_list: List[ImageT]

    def __init__(self, image_list: List[ImageT]):
        self.image_list = image_list

    def __len__(self) -> int:
        return len(self.image_list)

    def __getitem__(self, idx) -> torch.Tensor:
        return self.image_list[idx].to_tensor()

    @classmethod
    def from_chunk(
            cls: Type[AutoEncoderDataestT],
            chunk: MultiEpisodeChunk,
            n_augmentation: int = 2
    ) -> AutoEncoderDataestT:

        image_list: List[ImageT] = []
        for episode_data in chunk:
            image_seq: ElementSequence[ImageT] = episode_data.filter_by_type(cls.image_type)
            image_list.extend(image_seq)
        auged_image_list = cls.augmentation(image_list, n_augmentation)
        return cls(auged_image_list)

    @staticmethod
    @abstractmethod
    def augmentation(image_list: List[ImageT], n: int) -> List[ImageT]:
        pass


class RGBAutoEncoderDataset(AutoEncoderDataset[RGBImage]):
    image_type = RGBImage

    @staticmethod
    def augmentation(image_list: List[ImageT], n: int) -> List[ImageT]:
        aug_guass = al.GaussNoise(p=1)
        aug_rgbshit = al.RGBShift(r_shift_limit=40, g_shift_limit=40, b_shift_limit=40)
        aug_composed = al.Compose([aug_guass, aug_rgbshit])

        auged_list = []
        for i in range(n):
            auged_list.extend([aug_composed(image=img)['image'] for img in image_list])
        image_list.extend(auged_list)

        return image_list


class AutoRegressiveDataset(Dataset):
    state_seq_list: List[np.ndarray]

    def __init__(self, state_seq_list: List[np.ndarray]):
        self.state_seq_list = self.attach_flag_info(state_seq_list)

    def __len__(self) -> int:
        return len(self.state_seq_list)

    def __getitem__(self, idx) -> torch.Tensor:
        return torch.from_numpy(self.state_seq_list[idx]).float()

    @classmethod
    def from_chunk(cls, chunk: MultiEpisodeChunk, embed_rule: EmbeddingRule) -> 'AutoRegressiveDataset':
        state_seq_list = embed_rule.apply_to_multi_episode_chunk(chunk)
        return cls(state_seq_list)

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
