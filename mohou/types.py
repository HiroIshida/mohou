import numpy as np
from dataclasses import dataclass
from typing import Generic, List, Optional, Tuple, TypeVar, Callable, Type


class ElementBase:
    data: np.ndarray

    def __init__(self, data): self.data = data

    def shape(self): return self.data.shape

class AngleVector(ElementBase): ...

class ImageBase(ElementBase): ...

class RGBImage(ImageBase): ...

class DepthImage(ImageBase): ...

class RGBDImage(ImageBase): ...

DataT = TypeVar('DataT', bound=ElementBase)

@dataclass
class DataSequence(Generic[DataT]):
    data: List[DataT]
    def __len__(self): return len(self.data)
    def __getitem__(self, indices): return self.data[indices]

class SingleEpisodeData:
    types: List[Type] # https://docs.python.org/3/library/typing.html#typing.Type
    sequence_list: List[DataSequence]
    def __init__(self, sequence_list: List[DataSequence]):

        all_same_length = len(set(map(len, sequence_list))) == 1
        assert all_same_length

        types = set(map(lambda seq: type(seq[0]), sequence_list))
        n_type = len(types)
        all_different_type = n_type == len(sequence_list)
        assert all_different_type

        no_image_type_conflict = len([isinstance(seq[0], ImageBase) for seq in sequence_list]) == 1
        assert no_image_type_conflict

        self.types = list(types)
        self.sequence_list = sequence_list

    def filter_by_type(self, t: Type[DataT]) -> DataSequence[DataT]:
        for seq in self.sequence_list:
            if isinstance(seq[0], t):
                # thanks to all_different_type
                return seq
        assert False

@dataclass
class MultiEpisodeDataChunk:
    sedata_list: List[SingleEpisodeData]

