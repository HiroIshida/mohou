import functools
import operator
from typing import Generic, List, Tuple, Type, TypeVar, Iterator, Dict

import numpy as np
import torch
import torchvision

from mohou.file import load_object


class ElementBase(np.ndarray):
    def __new__(cls, arr):
        return np.asarray(arr).view(cls)

    def to_tensor(self) -> torch.Tensor:
        assert False


class VectorBase(ElementBase):

    def to_tensor(self) -> torch.Tensor:
        return torch.from_numpy(self).float()


class AngleVector(VectorBase):
    pass


class ImageBase(ElementBase):
    pass


class RGBImage(ImageBase):

    def to_tensor(self) -> torch.Tensor:
        return torchvision.transforms.ToTensor()(self).float()


class DepthImage(ImageBase):
    pass


class RGBDImage(ImageBase):
    pass


ElementT = TypeVar('ElementT', bound=ElementBase)
ImageT = TypeVar('ImageT', bound=ImageBase)


class ElementSequence(list, Generic[ElementT]):
    # TODO(HiroIshida) make it custom list

    @property
    def element_shape(self):
        return self.__getitem__(0).shape


class EpisodeData:
    types: List[Type[ElementBase]]
    type_shape_table: Dict[Type[ElementBase], Tuple[int, ...]]
    sequence_list: Tuple[ElementSequence, ...]

    def __init__(self, sequence_tuple: Tuple[ElementSequence, ...]):
        for sequence in sequence_tuple:
            assert isinstance(sequence, ElementSequence)

        all_same_length = len(set(map(len, sequence_tuple))) == 1
        assert all_same_length

        types = list(map(lambda seq: type(seq[0]), sequence_tuple))
        shapes = list(map(lambda seq: seq[0].shape, sequence_tuple))

        n_type = len(types)
        all_different_type = n_type == len(sequence_tuple)
        assert all_different_type

        # e.g. Having RGBImage and RGBDImage at the same time is not arrowed
        no_image_type_conflict = sum([isinstance(seq[0], ImageBase) for seq in sequence_tuple]) == 1
        assert no_image_type_conflict

        self.types = types
        self.type_shape_table = {t: s for (t, s) in zip(types, shapes)}
        self.sequence_list = sequence_tuple

    def filter_by_type(self, t: Type[ElementT]) -> ElementSequence[ElementT]:
        for seq in self.sequence_list:
            if isinstance(seq[0], t):
                # thanks to all_different_type
                return seq
        assert False

    def __iter__(self):
        return self.sequence_list.__iter__()


class MultiEpisodeChunk:
    data_list: List[EpisodeData]
    types: List[Type[ElementBase]]
    type_shape_table: Dict[Type[ElementBase], Tuple[int, ...]]

    def __init__(self, data_list: List[EpisodeData]):
        types = data_list[0].types
        n_type_appeared = len(set(functools.reduce(operator.add, [d.types for d in data_list])))
        assert n_type_appeared == len(types)

        self.data_list = data_list
        self.types = types
        self.type_shape_table = data_list[0].type_shape_table

    def __iter__(self) -> Iterator:
        return self.data_list.__iter__()

    def __getitem__(self, index):
        return self.data_list.__getitem__(index)

    def get_element_shape(self, elem_type: Type[ElementBase]) -> Tuple[int, ...]:
        return self.type_shape_table[elem_type]

    @classmethod
    def load(cls, project_name: str) -> 'MultiEpisodeChunk':
        return load_object(cls, project_name)
