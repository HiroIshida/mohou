import numpy as np
import torch
import torchvision
from dataclasses import dataclass
from typing import Generic, List, Optional, Tuple, TypeVar, Callable, Type, NewType


class ElementBase(np.ndarray):
    def __new__(cls, arr): return np.asarray(arr).view(cls)

    def to_tensor(self) -> torch.Tensor : assert False

class AngleVector(ElementBase):

    def to_tensor(self) -> torch.Tensor: return torch.from_numpy(self).float()

class ImageBase(ElementBase): ...

class RGBImage(ImageBase):

    def to_tensor(self) -> torch.Tensor:
        return torchvision.transforms.ToTensor()(self).float()

class DepthImage(ImageBase): ...

class RGBDImage(ImageBase): ...

ElementT = TypeVar('ElementT', bound=ElementBase)

class ElementSequence(list, Generic[ElementT]): ...

class SingleEpisodeData:
    types: List[Type] # https://docs.python.org/3/library/typing.html#typing.Type
    sequence_list: Tuple[ElementSequence, ...]
    def __init__(self, sequence_tuple: Tuple[ElementSequence, ...]):

        all_same_length = len(set(map(len, sequence_tuple))) == 1
        assert all_same_length

        types = set(map(lambda seq: type(seq[0]), sequence_tuple))
        n_type = len(types)
        all_different_type = n_type == len(sequence_tuple)
        assert all_different_type

        # e.g. Having RGBImage and RGBDImage at the same time is not arrowed
        no_image_type_conflict = sum([isinstance(seq[0], ImageBase) for seq in sequence_tuple]) == 1
        assert no_image_type_conflict

        self.types = list(types)
        self.sequence_list = sequence_tuple

    def filter_by_type(self, t: Type[ElementT]) -> ElementSequence[ElementT]:
        for seq in self.sequence_list:
            if isinstance(seq[0], t):
                # thanks to all_different_type
                return seq
        assert False

class MultiEpisodeDataChunk(List[SingleEpisodeData]): ...
