import numpy as np
from torch.utils.data import Dataset
from itertools import chain

from typing import List

from mohou.types import MultiEpisodeDataChunk
from mohou.types import ImageBase
from mohou.types import ElementSequence

class AutoEncoderDataset(Dataset):
    image_list: List[ImageBase]

    def __init__(self, image_list: List[ImageBase]):
        self.image_list = image_list

    @classmethod
    def from_chunk(cls, chunk: MultiEpisodeDataChunk):
        #image_seq_list: List[ImageBase] = [sedata.filter_by_type(ImageBase) for sedata in chunk.sedata_list]
        image_list: List[ImageBase] = []
        for sedata in chunk:
            image_seq: ElementSequence[ImageBase] = sedata.filter_by_type(ImageBase)
            image_list.extend(image_seq)

        return AutoEncoderDataset(image_list)
