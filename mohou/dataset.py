from typing import List

from torch.utils.data import Dataset

from mohou.types import ElementSequence, ImageBase, MultiEpisodeDataChunk


class AutoEncoderDataset(Dataset):
    image_list: List[ImageBase]

    def __init__(self, image_list: List[ImageBase]):
        self.image_list = image_list

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        return self.image_list[idx].to_tensor()

    @classmethod
    def from_chunk(cls, chunk: MultiEpisodeDataChunk):
        image_list: List[ImageBase] = []
        for sedata in chunk:
            image_seq: ElementSequence[ImageBase] = sedata.filter_by_type(ImageBase)
            image_list.extend(image_seq)

        return cls(image_list)
