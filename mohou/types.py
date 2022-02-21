from abc import ABC, abstractmethod, abstractclassmethod
import functools
import operator
import random
from typing import Generic, List, Tuple, Type, TypeVar, Iterator, Dict, Sequence, ClassVar

import numpy as np
import torch
import torchvision

from mohou.constant import N_DATA_INTACT
from mohou.file import load_object
from mohou.image_randomizer import _f_randomize_rgb_image, _f_randomize_depth_image

ElementT = TypeVar('ElementT', bound='ElementBase')
ImageT = TypeVar('ImageT', bound='ImageBase')
VectorT = TypeVar('VectorT', bound='VectorBase')


class ElementBase(ABC):

    @classmethod
    def is_concrete_type(cls):
        return len(cls.__abstractmethods__) == 0

    @abstractmethod
    def shape(self) -> Tuple[int, ...]:
        pass

    @abstractmethod
    def to_tensor(self) -> torch.Tensor:
        pass

    @abstractclassmethod
    def from_tensor(cls, tensor: torch.Tensor) -> 'ElementBase':
        pass


class VectorBase(np.ndarray, ElementBase):

    def __new__(cls, arr):
        return np.asarray(arr).view(cls)

    def to_tensor(self) -> torch.Tensor:
        return torch.from_numpy(self).float()

    @classmethod
    def from_tensor(cls: Type[VectorT], tensor: torch.Tensor) -> VectorT:
        array = tensor.detach().clone().numpy()
        return cls(array)


class AngleVector(VectorBase):
    pass


class ImageBase(np.ndarray, ElementBase):
    channel: ClassVar[int]

    def __new__(cls, arr):
        return np.asarray(arr).view(cls)

    @abstractmethod
    def randomize(self) -> 'ImageBase':
        pass


class RGBImage(ImageBase):
    channel: ClassVar[int] = 3

    def to_tensor(self) -> torch.Tensor:
        return torchvision.transforms.ToTensor()(self).float()

    @classmethod
    def from_tensor(cls, tensor: torch.Tensor) -> 'RGBImage':
        tf = torchvision.transforms.ToPILImage()
        pil_iamge = tf(tensor)
        return cls(pil_iamge)

    def randomize(self) -> 'RGBImage':
        assert _f_randomize_rgb_image is not None
        rand_image_arr = _f_randomize_rgb_image(self)
        return RGBImage(rand_image_arr)


class DepthImage(ImageBase):
    channel: ClassVar[int] = 1

    def to_tensor(self) -> torch.Tensor:
        return torch.from_numpy(self).float()

    @classmethod
    def from_tensor(cls, tensor: torch.Tensor) -> 'DepthImage':
        array = tensor.detach().clone().numpy()
        return cls(array)

    def randomize(self) -> 'DepthImage':
        assert _f_randomize_depth_image is not None
        rand_depth_arr = _f_randomize_depth_image(self)
        return DepthImage(rand_depth_arr)


class ElementDict(Dict[Type[ElementBase], ElementBase]):

    def __init__(self, elems: Sequence[ElementBase]):
        for elem in elems:
            self[elem.__class__] = elem
        assert len(set(self.keys())) == len(elems)

    def __getitem__(self, key: Type[ElementT]) -> ElementT:
        return super().__getitem__(key)  # type: ignore


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

        types = set(map(lambda seq: type(seq[0]), sequence_tuple))
        shapes = list(map(lambda seq: seq[0].shape, sequence_tuple))

        n_type = len(types)
        all_different_type = n_type == len(sequence_tuple)
        assert all_different_type, 'all sequences must have different type'

        self.types = list(types)
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
    data_list_intact: List[EpisodeData]
    types: List[Type[ElementBase]]
    type_shape_table: Dict[Type[ElementBase], Tuple[int, ...]]

    def __init__(
            self, data_list: List[EpisodeData],
            shuffle: bool = True, with_intact_data: bool = True):

        types = data_list[0].types
        n_type_appeared = len(set(functools.reduce(operator.add, [d.types for d in data_list])))
        assert n_type_appeared == len(types)

        if shuffle:
            random.shuffle(data_list)

        if with_intact_data:
            self.data_list_intact = data_list[:N_DATA_INTACT]
            self.data_list = data_list[N_DATA_INTACT:]
        else:
            self.data_list_intact = []
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

    def get_intact_chunk(self) -> 'MultiEpisodeChunk':
        return MultiEpisodeChunk(self.data_list_intact, shuffle=False, with_intact_data=False)
