from abc import ABC, abstractmethod
import copy
import functools
import operator
import random
from typing import Generic, List, Tuple, Type, TypeVar, Iterator, Sequence, ClassVar, OrderedDict

import numpy as np
import torch
import torchvision

from mohou.constant import N_DATA_INTACT
from mohou.file import load_object
from mohou.image_randomizer import _f_randomize_rgb_image, _f_randomize_depth_image
from mohou.utils import split_sequence

ElementT = TypeVar('ElementT', bound='ElementBase')
SingleElementT = TypeVar('SingleElementT', bound='SingleElementBase')
SingleImageT = TypeVar('SingleImageT', bound='SingleImageBase')
MixedImageT = TypeVar('MixedImageT', bound='MixedImageBase')
ImageT = TypeVar('ImageT', bound='ImageBase')
VectorT = TypeVar('VectorT', bound='VectorBase')


class ElementBase(ABC):

    @classmethod
    def is_concrete_type(cls):
        return len(cls.__abstractmethods__) == 0 and len(cls.__subclasses__()) == 0

    @abstractmethod
    def shape(self) -> Tuple[int, ...]:
        pass

    @abstractmethod
    def to_tensor(self) -> torch.Tensor:
        pass

    @classmethod
    @abstractmethod
    def from_tensor(cls: Type[ElementT], tensor: torch.Tensor) -> ElementT:
        pass


class SingleElementBase(np.ndarray, ElementBase):

    def __new__(cls, arr):
        # instantiationg blocking hack. Different but similar to
        # https://stackoverflow.com/a/7990308/7624196
        assert cls.is_concrete_type(),\
            '{} is an abstract class and thus cannot instantiate'.format(cls.__name__)
        return np.asarray(arr).view(cls)


class VectorBase(SingleElementBase):

    def to_tensor(self) -> torch.Tensor:
        return torch.from_numpy(self).float()

    @classmethod
    def from_tensor(cls: Type[VectorT], tensor: torch.Tensor) -> VectorT:
        array = tensor.detach().clone().numpy()
        return cls(array)


class AngleVector(VectorBase):
    pass


class ImageBase(ElementBase):

    @classmethod
    @abstractmethod
    def channel(cls) -> int:
        pass

    @abstractmethod
    def randomize(self: ImageT) -> ImageT:
        pass

    @classmethod
    @abstractmethod
    def dummy_from_shape(cls: Type[ImageT], shape2d: Tuple[int, int]) -> ImageT:
        # TODO(HiroIshida) I'm currently asking here
        # https://stackoverflow.com/questions/71214808/
        pass


class SingleImageBase(SingleElementBase, ImageBase):
    _channel: ClassVar[int]

    @classmethod
    def channel(cls) -> int:
        return cls._channel


class RGBImage(SingleImageBase):
    _channel: ClassVar[int] = 3

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

    @classmethod
    def dummy_from_shape(cls, shape2d: Tuple[int, int]) -> 'RGBImage':
        shape = (shape2d[0], shape2d[1], cls.channel())
        dummy_array = np.random.randint(0, high=255, size=shape, dtype=np.uint8)
        return cls(dummy_array)


class DepthImage(SingleImageBase):
    _channel: ClassVar[int] = 1

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

    @classmethod
    def dummy_from_shape(cls, shape2d: Tuple[int, int]) -> 'DepthImage':
        shape = (shape2d[0], shape2d[1], cls.channel())
        dummy_array = np.random.randn(*shape)
        return cls(dummy_array)


class MixedImageBase(ImageBase):
    image_types: ClassVar[List[Type[SingleImageBase]]]
    images: List[SingleImageBase]
    _shape: Tuple[int, int, int]

    def __init__(self, images, check_size=False):

        image_shape = images.shape[:2]

        if check_size:
            for image in images:
                assert image.shape[:2] == image_shape

        self.images = images
        self._shape = (image_shape[0], image_shape[1], self.channel())

    def to_tensor(self) -> torch.Tensor:
        return torch.cat([im.to_tensor() for im in self.images])

    def shape(self) -> Tuple[int, int, int]:
        return self._shape

    def randomize(self: MixedImageT) -> MixedImageT:
        rand = copy.deepcopy(self)
        for i in range(len(rand.images)):
            rand.images[i] = self.images[i].randomize()
        return rand

    @classmethod
    def channel(cls) -> int:
        return sum([t.channel() for t in cls.image_types])

    @classmethod
    def from_tensor(cls: Type[MixedImageT], tensor: torch.Tensor) -> MixedImageT:
        channel_list = [t.channel() for t in cls.image_types]
        images = list(split_sequence(tensor, channel_list))
        return cls(images)

    @classmethod
    def dummy_from_shape(cls: Type[MixedImageT], shape2d: Tuple[int, int]) -> MixedImageT:
        images = [t.dummy_from_shape(shape2d) for t in cls.image_types]
        return cls(images)

    def get_single_image(self, image_type: Type[SingleImageT]) -> SingleImageT:
        for image in self.images:
            if isinstance(image, image_type):
                return image
        assert False


class RGBDImage(MixedImageBase):
    image_types = [RGBImage, DepthImage]


class ElementDict(OrderedDict[Type[ElementBase], ElementBase]):

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


def create_mixed_image_sequence(mixed_image_type: MixedImageT, elem_seqs: List[ElementSequence]) -> ElementSequence[MixedImageT]:
    # TODO(HiroIshida) extend this to 'mixed_element_sequence'
    n_len_seq = len(elem_seqs[0])
    mixed_image_seq = ElementSequence[MixedImageT]([])
    for i in range(n_len_seq):
        mixed_image = mixed_image_type.__init__([seq[i] for seq in elem_seqs])  # type: ignore
        mixed_image_seq.append(mixed_image)
    return mixed_image_seq


class EpisodeData:
    type_shape_table: OrderedDict[Type[ElementBase], Tuple[int, ...]]
    sequence_list: Tuple[ElementSequence, ...]

    def __init__(self, sequence_tuple: Tuple[ElementSequence, ...]):
        for sequence in sequence_tuple:
            assert isinstance(sequence, ElementSequence)

        all_same_length = len(set(map(len, sequence_tuple))) == 1
        assert all_same_length

        types = [type(seq[0]) for seq in sequence_tuple]
        shapes = [seq[0].shape for seq in sequence_tuple]
        type_shape_table = OrderedDict({t: s for (t, s) in zip(types, shapes)})

        n_type = len(set(types))
        all_different_type = n_type == len(sequence_tuple)
        assert all_different_type, 'all sequences must have different type'

        self.type_shape_table = type_shape_table
        self.sequence_list = sequence_tuple

    def filter_by_single_type(self, elem_type: Type[SingleElementT]) -> ElementSequence[SingleElementT]:
        for seq in self.sequence_list:
            if isinstance(seq[0], elem_type):
                # thanks to all_different_type
                return seq
        assert False

    def filter_by_type(self, elem_type: Type[ElementT]) -> ElementSequence[ElementT]:

        if issubclass(elem_type, SingleElementBase):
            return self.filter_by_single_type(elem_type)  # type: ignore
        elif issubclass(elem_type, MixedImageBase):
            seqs = [self.filter_by_single_type(t) for t in elem_type.image_types]
            return create_mixed_image_sequence(elem_type, seqs)  # type: ignore
        else:
            assert False

    def __iter__(self):
        return self.sequence_list.__iter__()


class MultiEpisodeChunk:
    data_list: List[EpisodeData]
    data_list_intact: List[EpisodeData]
    type_shape_table: OrderedDict[Type[ElementBase], Tuple[int, ...]]

    def __init__(
            self, data_list: List[EpisodeData],
            shuffle: bool = True, with_intact_data: bool = True):

        set_types = set(functools.reduce(
            operator.add,
            [list(data.type_shape_table.keys()) for data in data_list]))

        n_type_appeared = len(set_types)
        assert n_type_appeared == len(data_list[0].type_shape_table.keys())

        if shuffle:
            random.shuffle(data_list)

        if with_intact_data:
            self.data_list_intact = data_list[:N_DATA_INTACT]
            self.data_list = data_list[N_DATA_INTACT:]
        else:
            self.data_list_intact = []
            self.data_list = data_list

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
