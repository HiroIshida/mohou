from typing import List

from torch.utils.data import Dataset

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
    def from_chunk(cls, chunk: MultiEpisodeChunk):
        image_list: List[ImageBase] = []
        for episode_data in chunk:
            image_seq: ElementSequence[ImageBase] = episode_data.filter_by_type(ImageBase)
            image_list.extend(image_seq)

        return cls(image_list)
