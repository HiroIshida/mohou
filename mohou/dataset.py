import numpy as np
from typing import List

from torch.utils.data import Dataset

from mohou.constant import CONTINUE_FLAG_VALUE, END_FLAG_VALUE
from mohou.embedding_rule import EmbeddingRule
from mohou.types import ElementSequence, ImageBase, MultiEpisodeChunk


class AutoEncoderDataset(Dataset):
    image_list: List[ImageBase]

    def __init__(self, image_list: List[ImageBase]):
        self.image_list = image_list

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        return self.image_list[idx].to_tensor()

    @classmethod
    def from_chunk(cls, chunk: MultiEpisodeChunk) -> 'AutoEncoderDataset':
        image_list: List[ImageBase] = []
        for episode_data in chunk:
            image_seq: ElementSequence[ImageBase] = episode_data.filter_by_type(ImageBase)
            image_list.extend(image_seq)

        return cls(image_list)


class AutoRegressiveDataset(Dataset):
    state_seq_list: List[np.ndarray]

    def __init__(self, state_seq_list: List[np.ndarray]):
        self.state_seq_list = self.attach_flag_info(state_seq_list)

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

            flag_seq = np.ones(n_max_in_dataset)
            flag_seq[:n_seq] *= CONTINUE_FLAG_VALUE
            flag_seq[n_seq:] *= END_FLAG_VALUE

            padding_state_seq = np.tile(state_seq[-1], (n_padding, 1))
            padded_state_seq = np.vstack((state_seq, padding_state_seq))
            padded_state_flag_seq = np.hstack((padded_state_seq, flag_seq))

            state_seq_list[i] = padded_state_flag_seq

        return state_seq_list
