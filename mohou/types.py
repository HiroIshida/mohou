import copy
import functools
import hashlib
import json
import logging
import operator
import os
import pathlib
import pickle
import random
import re
import subprocess
import tempfile
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import (
    ClassVar,
    Dict,
    Generic,
    Iterable,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
    overload,
)

import cv2
import matplotlib.pyplot as plt
import natsort
import numpy as np
import PIL.Image
import torch
import torchvision
import yaml
from moviepy.editor import ImageSequenceClip

from mohou.constant import CONTINUE_FLAG_VALUE, TERMINATE_FLAG_VALUE
from mohou.image_randomizer import (
    _f_randomize_depth_image,
    _f_randomize_gray_image,
    _f_randomize_rgb_image,
)
from mohou.utils import (
    assert_equal_with_message,
    assert_isinstance_with_message,
    canvas_to_ndarray,
    get_all_concrete_leaftypes,
    split_sequence,
)

logger = logging.getLogger(__name__)

ElementT = TypeVar("ElementT", bound="ElementBase")
PrimitiveElementT = TypeVar("PrimitiveElementT", bound="PrimitiveElementBase")
PrimitiveImageT = TypeVar("PrimitiveImageT", bound="PrimitiveImageBase")
ColorImageT = TypeVar("ColorImageT", bound="ColorImageBase")
CompositeImageT = TypeVar("CompositeImageT", bound="CompositeImageBase")
ImageT = TypeVar("ImageT", bound="ImageBase")
VectorT = TypeVar("VectorT", bound="VectorBase")


CompositeListElementT = TypeVar("CompositeListElementT")


class Hashable:
    @property
    def hash_value(self) -> str:
        data_pickle = pickle.dumps(self)
        data_md5 = hashlib.md5(data_pickle).hexdigest()
        return data_md5[:8]


class MetaData(Dict[str, Union[str, int, float]], Hashable):
    pass


class HasAList(Generic[CompositeListElementT]):
    @abstractmethod
    def _get_has_a_list(self) -> List[CompositeListElementT]:
        pass

    def __iter__(self) -> Iterator[CompositeListElementT]:
        return self._get_has_a_list().__iter__()

    @overload
    def __getitem__(self, indices: List[int]) -> List[CompositeListElementT]:
        pass

    @overload
    def __getitem__(self, indices: slice) -> List[CompositeListElementT]:
        pass

    @overload
    def __getitem__(self, index: int) -> CompositeListElementT:
        pass

    def __getitem__(self, indices_like):  # TODO(HiroIshida) add type hints?
        lst_inner = self._get_has_a_list()
        if isinstance(indices_like, list):
            return [lst_inner[i] for i in indices_like]
        else:
            return lst_inner[indices_like]

    def __len__(self) -> int:
        return len(self._get_has_a_list())


class ElementBase(ABC):
    def __new__(cls, *args, **kwargs):
        # instantiationg blocking hack. Different but similar to
        # https://stackoverflow.com/a/7990308/7624196
        assert cls.is_concrete_type(), "{} is an abstract class and thus cannot instantiate".format(
            cls.__name__
        )
        # https://stackoverflow.com/questions/59217884/
        return super(ElementBase, cls).__new__(cls)

    @classmethod
    def is_concrete_type(cls):
        return len(cls.__abstractmethods__) == 0 and len(cls.__subclasses__()) == 0  # type: ignore [attr-defined]

    @property
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


class PrimitiveElementBase(ElementBase):
    _data: np.ndarray  # cmposition over inheritance!

    def __init__(self, data: np.ndarray) -> None:
        assert_isinstance_with_message(data, np.ndarray)
        assert not np.isnan(data).any()
        assert not np.isinf(data).any()
        self._data = np.array(data)

    @property
    def shape(self) -> Tuple[int, ...]:
        return self._data.shape

    def numpy(self):
        return self._data

    def __iter__(self) -> Iterator:
        return self._data.__iter__()

    def __getitem__(self, index):
        return self._data[index]

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PrimitiveElementBase):
            return NotImplemented
        assert type(self) == type(other)
        return np.allclose(self._data, other._data, atol=1e-6)


class VectorBase(PrimitiveElementBase):
    def __init__(self, data: np.ndarray) -> None:
        super().__init__(data)
        assert self._data.ndim == 1

    def to_tensor(self) -> torch.Tensor:
        return torch.from_numpy(self._data).float()

    @classmethod
    def from_tensor(cls: Type[VectorT], tensor: torch.Tensor) -> VectorT:
        array = tensor.detach().clone().numpy()
        return cls(array)

    def __len__(self):
        return len(self._data)


class AngleVector(VectorBase):
    pass


class GripperState(VectorBase):
    pass


class AnotherGripperState(VectorBase):
    # I know naming RarmGripperState and LarmGripperState is cleaner.
    # But for the backward-compatibility, we cannot change type name of
    # GripperState
    pass


class TerminateFlag(VectorBase):
    @classmethod
    def from_bool(cls, flag: bool) -> "TerminateFlag":
        assert isinstance(flag, bool)
        val = TERMINATE_FLAG_VALUE if flag else CONTINUE_FLAG_VALUE
        data = np.array([val], dtype=np.float64)
        return cls(data)


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
        pass

    @abstractmethod
    def to_rgb(self, *args, **kwargs) -> "RGBImage":
        pass

    @abstractmethod
    def resize(self, shape2d_new: Tuple[int, int]) -> None:
        pass


class PrimitiveImageBase(PrimitiveElementBase, ImageBase):
    _channel: ClassVar[int]

    def __init__(self, data: np.ndarray) -> None:
        super().__init__(data)
        assert_equal_with_message(self._data.ndim, 3, "image_dim")
        assert_equal_with_message(data.shape[2], self.channel(), "channel")

    @classmethod
    def channel(cls) -> int:
        return cls._channel


class ColorImageBase(PrimitiveImageBase, Generic[ColorImageT]):
    def __init__(self, data: np.ndarray) -> None:
        super().__init__(data)
        assert_equal_with_message(self._data.dtype.type, np.uint8, "numpy type")

    def to_tensor(self) -> torch.Tensor:
        return torchvision.transforms.ToTensor()(self._data).float()

    @classmethod
    def dummy_from_shape(cls: Type[ColorImageT], shape2d: Tuple[int, int]) -> ColorImageT:
        shape = (shape2d[0], shape2d[1], cls.channel())
        dummy_array = np.random.randint(0, high=255, size=shape, dtype=np.uint8)
        return cls(dummy_array)


class RGBImage(ColorImageBase["RGBImage"]):
    _channel: ClassVar[int] = 3

    @classmethod
    def from_file(cls, filename: str) -> "RGBImage":
        pil_img = PIL.Image.open(filename).convert("RGB")
        arr = np.array(pil_img)
        return cls(arr)

    @classmethod
    def from_tensor(cls, tensor: torch.Tensor) -> "RGBImage":
        tf = torchvision.transforms.ToPILImage()
        pil_iamge = tf(tensor)
        return cls(np.array(pil_iamge))

    def randomize(self) -> "RGBImage":
        assert _f_randomize_rgb_image is not None
        rand_image_arr = _f_randomize_rgb_image(self._data)
        return RGBImage(rand_image_arr)

    def to_rgb(self, *args, **kwargs) -> "RGBImage":
        return self

    def resize(self, shape2d_new: Tuple[int, int]) -> None:
        self._data = cv2.resize(self._data, shape2d_new, interpolation=cv2.INTER_AREA)

    def bgr2rgb(self) -> "RGBImage":
        data_new = self._data[..., ::-1].copy()
        return RGBImage(data_new)


class GrayImage(ColorImageBase["GrayImage"]):
    _channel: ClassVar[int] = 1

    @classmethod
    def from_tensor(cls, tensor: torch.Tensor) -> "GrayImage":
        tensor2d = tensor.squeeze(dim=0)
        tf = torchvision.transforms.ToPILImage()
        pil_iamge = tf(tensor2d)
        return cls(np.expand_dims(np.array(pil_iamge), axis=2))

    def randomize(self) -> "GrayImage":
        assert _f_randomize_gray_image is not None
        rand_image_arr = _f_randomize_gray_image(self._data)
        return GrayImage(rand_image_arr)

    def to_rgb(self, *args, **kwargs) -> RGBImage:
        arr = np.array(cv2.cvtColor(self._data[:, :, 0], cv2.COLOR_GRAY2RGB))
        return RGBImage(arr)

    def resize(self, shape2d_new: Tuple[int, int]) -> None:
        arr = cv2.resize(self._data, shape2d_new, interpolation=cv2.INTER_AREA)
        self._data = np.expand_dims(arr, axis=2)


def extract_contour_by_laplacian(
    rgb: RGBImage, laplace_kernel_size: int = 3, blur_kernel_size: Optional[Tuple[int, int]] = None
) -> GrayImage:

    if blur_kernel_size is None:
        blur_kernel_size = (int(rgb.shape[0] * 0.02), int(rgb.shape[1] * 0.02))

    src_gray = cv2.cvtColor(rgb.numpy(), cv2.COLOR_BGR2GRAY)
    dst = cv2.Laplacian(src_gray, cv2.CV_8U, ksize=laplace_kernel_size)
    dst2 = cv2.blur(dst, (blur_kernel_size[0], blur_kernel_size[1]))
    return GrayImage(np.expand_dims(np.uint8(dst2), axis=2))


class DepthImage(PrimitiveImageBase):
    _channel: ClassVar[int] = 1
    _max_value: ClassVar[float] = 4.0
    _min_value: ClassVar[float] = -1.0

    def __init__(self, data: np.ndarray) -> None:
        super().__init__(data)
        assert_equal_with_message(
            self._data.dtype.type, [np.float16, np.float32, np.float64], "numpy type"
        )

    def to_tensor(self) -> torch.Tensor:
        data_cutoff = np.maximum(np.minimum(self._data, self._max_value), self._min_value)
        data_normalized = (data_cutoff - self._min_value) / (self._max_value - self._min_value)
        return torch.from_numpy(data_normalized.transpose((2, 0, 1))).contiguous().float()

    @classmethod
    def from_tensor(cls, tensor: torch.Tensor) -> "DepthImage":
        data = tensor.detach().clone().numpy().transpose((1, 2, 0))
        data_denormalized = data * (cls._max_value - cls._min_value) + cls._min_value
        return cls(data_denormalized)

    def randomize(self) -> "DepthImage":
        assert _f_randomize_depth_image is not None
        rand_depth_arr = _f_randomize_depth_image(self._data)
        return DepthImage(rand_depth_arr)

    @classmethod
    def dummy_from_shape(cls, shape2d: Tuple[int, int]) -> "DepthImage":
        shape = (shape2d[0], shape2d[1], cls.channel())
        dummy_array = np.random.rand(*shape)
        return cls(dummy_array)

    def to_rgb(self, *args, **kwargs) -> "RGBImage":
        fig = plt.figure()
        ax = plt.subplot(1, 1, 1)
        ax.imshow(self._data[:, :, 0])
        fig.canvas.draw()
        arr = canvas_to_ndarray(fig)
        plt.close(fig)
        return RGBImage(arr)

    def resize(self, shape2d_new: Tuple[int, int]) -> None:
        tmp = cv2.resize(self._data, shape2d_new, interpolation=cv2.INTER_CUBIC)
        self._data = np.expand_dims(tmp, axis=2)


class CompositeImageBase(ImageBase):
    image_types: ClassVar[List[Type[PrimitiveImageBase]]]
    images: List[PrimitiveImageBase]

    def __init__(self, images: List[PrimitiveImageBase], check_size=True):

        image_shape = images[0].shape[:2]

        if check_size:
            for image in images:
                assert_equal_with_message(image.shape[:2], image_shape, "image w-h")

        for image, image_type in zip(images, self.image_types):
            assert_isinstance_with_message(image, image_type)

        self.images = images

    def to_tensor(self) -> torch.Tensor:
        return torch.cat([im.to_tensor() for im in self.images])

    @property
    def shape(self) -> Tuple[int, int, int]:
        shape_2d = self.images[0].shape[:2]
        return (shape_2d[0], shape_2d[1], self.channel())

    def randomize(self: CompositeImageT) -> CompositeImageT:
        rand = copy.deepcopy(self)
        for i in range(len(rand.images)):
            rand.images[i] = self.images[i].randomize()
        return rand

    @classmethod
    def channel(cls) -> int:
        return sum([t.channel() for t in cls.image_types])

    @classmethod
    def from_tensor(cls: Type[CompositeImageT], tensor: torch.Tensor) -> CompositeImageT:
        channel_list = [t.channel() for t in cls.image_types]
        images = []
        for image_type, sub_tensor in zip(cls.image_types, split_sequence(tensor, channel_list)):
            image = image_type.from_tensor(sub_tensor)
            images.append(image)
        return cls(images)

    @classmethod
    def dummy_from_shape(cls: Type[CompositeImageT], shape2d: Tuple[int, int]) -> CompositeImageT:
        images = [t.dummy_from_shape(shape2d) for t in cls.image_types]
        return cls(images)

    def get_primitive_image(self, image_type: Type[PrimitiveImageT]) -> PrimitiveImageT:
        for image in self.images:
            if isinstance(image, image_type):
                return image
        assert False

    def resize(self, shape2d_new: Tuple[int, int]) -> None:
        for image in self.images:
            image.resize(shape2d_new)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CompositeImageBase):
            return NotImplemented
        assert type(self) == type(other)
        for im_self, im_other in zip(self.images, other.images):
            if im_self != im_other:
                return False
        return True


class RGBDImage(CompositeImageBase):
    image_types = [RGBImage, DepthImage]

    def to_rgb(self, *args, **kwargs) -> RGBImage:
        for image in self.images:
            if isinstance(image, RGBImage):
                return image
        assert False


class ElementDict(Dict[Type[PrimitiveElementBase], PrimitiveElementBase]):
    def __init__(self, elems: Sequence[ElementBase]):
        for elem in elems:
            self.__setitem__(elem.__class__, elem)

    def get_subdict(self, keys: Iterable[Type[ElementBase]]) -> "ElementDict":
        return ElementDict([self.__getitem__(key) for key in keys])

    def __setitem__(self, elem_type: Type[ElementBase], elem: ElementBase):
        assert type(elem) == elem_type

        if isinstance(elem, CompositeImageBase):
            assert issubclass(elem_type, CompositeImageBase)
            for sub_elem_type, sub_elem in zip(elem.image_types, elem.images):
                super().__setitem__(sub_elem_type, sub_elem)
        elif isinstance(elem, PrimitiveElementBase):
            assert issubclass(elem_type, PrimitiveElementBase)
            super().__setitem__(elem_type, elem)
        else:
            assert False

    def __getitem__(self, key: Type[ElementT]) -> ElementT:
        if issubclass(key, PrimitiveElementBase):
            return super().__getitem__(key)  # type: ignore
        elif issubclass(key, CompositeImageBase):
            if key in self:
                return super().__getitem__(key)  # type: ignore

            # TODO(HiroIshida) somehow, if the following is written in comprehension
            # then we get "TypeError: super(type, obj): obj must be an instance or subtype of type"
            images = []
            for imt in key.image_types:
                images.append(super().__getitem__(imt))
            return key(images)  # type: ignore

        else:
            assert False

    def __eq__(self, other: object):
        if not isinstance(other, ElementDict):
            return NotImplemented
        assert isinstance(other, ElementDict)
        for key in self.keys():
            if self[key] != other[key]:
                return False
        return True


def get_element_type(type_name: str) -> Type[ElementBase]:
    for t in get_all_concrete_leaftypes(ElementBase):
        if type_name == t.__name__:
            return t
    assert False, "type {} not found".format(type_name)


@dataclass(frozen=True)
class ElementSequence(HasAList[ElementT], Generic[ElementT]):
    elem_list: List[ElementT]

    def __post_init__(self):
        # validation
        assert isinstance(self.elem_list, list)
        assert len(self.elem_list) > 0
        assert len(set([type(elem) for elem in self.elem_list])) == 1
        assert len(set([elem.shape for elem in self.elem_list])) == 1

    @property
    def elem_type(self) -> Type[ElementT]:
        return type(self.elem_list[0])

    @property
    def elem_shape(self) -> Tuple[int, ...]:
        return self.elem_list[0].shape

    def _get_has_a_list(self) -> List[ElementT]:
        return self.elem_list

    def append(self, *args, **kwargs):  # type: ignore
        raise NotImplementedError("deleted method")

    def dump(self, episode_dir_path: pathlib.Path, compress: bool = False) -> None:

        if compress and self.elem_type == RGBImage:
            mp4_file_path = episode_dir_path / "sequence-{}.mp4".format(RGBImage.__name__)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore
            writer = cv2.VideoWriter(str(mp4_file_path), fourcc, 20, tuple(self.elem_shape[:2]))  # type: ignore
            for elem in self.elem_list:
                writer.write(elem.numpy())  # type: ignore
            writer.release()
        else:
            file_path = episode_dir_path / "sequence-{}.npy".format(self.elem_type.__name__)
            assert issubclass(self.elem_type, PrimitiveElementBase)
            seq = np.array([e.numpy() for e in self.elem_list])  # type: ignore
            np.save(file_path, seq)

    @classmethod
    def load(
        cls, episode_dir_path: pathlib.Path, elem_type: Type[ElementT]
    ) -> "ElementSequence[ElementT]":

        if elem_type == RGBImage:
            mp4_file_path = episode_dir_path / "sequence-{}.mp4".format(RGBImage.__name__)
            if mp4_file_path.exists():
                cap = cv2.VideoCapture(str(mp4_file_path))  # type: ignore
                rgb_seq = []
                while True:
                    ret, frame = cap.read()
                    if ret:
                        rgb_image = RGBImage(frame)
                        rgb_seq.append(rgb_image)
                    else:
                        break
                return ElementSequence[RGBImage](rgb_seq)  # type: ignore [return-value]
            # else, just load from npy file

        file_path = episode_dir_path / "sequence-{}.npy".format(elem_type.__name__)
        seq = np.load(file_path)
        elem_list = [elem_type(arr) for arr in seq]
        return ElementSequence(elem_list)

    @classmethod
    def load_all(cls, episode_dir_path: pathlib.Path) -> "Dict[Type[ElementBase], ElementSequence]":
        d = {}
        for p in episode_dir_path.iterdir():
            result = re.match(r"sequence-(\w+).(\w+)", p.name)
            if result is not None:
                elem_type = get_element_type(result.group(1))
                d[elem_type] = cls.load(episode_dir_path, elem_type)  # type: ignore
        return d


def create_composite_image_sequence(
    composite_image_type: Type[CompositeImageT],
    elemseq_list: List[ElementSequence[PrimitiveImageBase]],
) -> ElementSequence[CompositeImageT]:
    """Create composite image ElementSequence from different-typed element sequence
    composite_image_type: e.g. RGBDImage
    elemseq_list: e.g. [RGBImage, DepthImage]
    """

    n_len_seq = len(elemseq_list[0])
    composite_image_list = []
    for i in range(n_len_seq):
        composite_image: CompositeImageT = composite_image_type([seq[i] for seq in elemseq_list])
        composite_image_list.append(composite_image)
    return ElementSequence(composite_image_list)


class HasTypeShapeTable:
    def types(self) -> List[Type[ElementBase]]:
        return list(self.type_shape_table.keys())  # type: ignore

    def get_element_shape(self, elem_type: Type[ElementBase]) -> Tuple[int, ...]:
        return self.type_shape_table[elem_type]  # type: ignore


@dataclass
class TimeStampSequence(HasAList[float]):
    """data must be increasing order"""

    _data: List[float]

    def __post_init__(self):
        assert sorted(self._data) == self._data

    def _get_has_a_list(self) -> List[float]:
        return self._data

    def index_geq(self, time: float) -> int:
        """find index greater than or equal to time"""
        eps = 1e-8
        return [i for i, t in enumerate(self._data) if t > time - eps][0]

    def dump(self, episode_dir_path: pathlib.Path) -> None:
        file_path = episode_dir_path / "time_stamp_sequence.npy"
        np.save(file_path, np.array(self._data))

    @classmethod
    def load(cls, episode_dir_path: pathlib.Path) -> Optional["TimeStampSequence"]:
        file_path = episode_dir_path / "time_stamp_sequence.npy"
        if file_path.exists():
            seq = np.load(file_path).tolist()
            return TimeStampSequence(seq)
        else:
            return None


@dataclass
class EpisodeData(HasTypeShapeTable, Hashable):
    sequence_dict: Dict[Type[ElementBase], ElementSequence[ElementBase]]
    metadata: MetaData
    time_stamp_seq: Optional[TimeStampSequence] = None

    @classmethod
    def validate_sequence_dict(
        cls,
        sequence_dict: Dict[Type[ElementBase], ElementSequence[ElementBase]],
        check_terminate_flag: bool = True,
    ) -> None:
        # call this after update
        lengths = [len(seq) for seq in sequence_dict.values()]
        all_same_length = len(set(lengths)) == 1
        assert all_same_length

        # check is sequence
        is_sequence = lengths[0] > 1
        assert is_sequence

        if check_terminate_flag:
            assert TerminateFlag in sequence_dict
            flag_seq: ElementSequence[TerminateFlag] = sequence_dict[TerminateFlag]  # type: ignore
            cls.validate_terminate_flags(flag_seq)

    @staticmethod
    def validate_terminate_flags(flag_seq: ElementSequence[TerminateFlag]) -> None:
        initial_flag, last_flag = flag_seq[0], flag_seq[-1]
        assert initial_flag.numpy().item() == CONTINUE_FLAG_VALUE
        assert last_flag.numpy().item() == TERMINATE_FLAG_VALUE

        # terminate flag must change from 0 to 1
        change_count = 0
        for i in range(len(flag_seq) - 1):
            if flag_seq[i + 1].numpy().item() != flag_seq[i].numpy().item():
                change_count += 1
        assert change_count == 1

    def __len__(self) -> int:
        return len(self.sequence_dict[TerminateFlag])

    @property
    def type_shape_table(self) -> Dict[Type[ElementBase], Tuple[int, ...]]:
        dic = {}
        for key, seq in self.sequence_dict.items():
            dic[key] = seq.elem_shape
        return dic

    @staticmethod
    def create_default_terminate_flag_seq(n_length) -> ElementSequence[TerminateFlag]:
        flag_lst = [TerminateFlag.from_bool(False) for _ in range(n_length - 1)]
        flag_lst.append(TerminateFlag.from_bool(True))
        elem_seq = ElementSequence(flag_lst)
        return elem_seq

    @classmethod
    def from_seq_list(
        cls,
        sequence_list: List[ElementSequence],
        timestamp_seq: Optional[TimeStampSequence] = None,
        metadata: Optional[MetaData] = None,
        check_terminate_flags: bool = True,
    ) -> "EpisodeData":
        r"""Create episode from list of ElementSequence
        Args:
            sequence_list: list of ElementSequence to construct the episode
            timestamp_seq: optional timestamp information
            metadata: optional metadata
            check_terminate_flag: if True, validation of TerminateFlag sequence will be  conducted. (see: validate_terminate_flags function)
        """

        if metadata is None:
            metadata = MetaData({})

        for sequence in sequence_list:
            assert isinstance(sequence, ElementSequence)

        if timestamp_seq is not None:
            assert len(sequence_list[0]) == len(timestamp_seq)

        types = [type(seq[0]) for seq in sequence_list]

        if TerminateFlag not in set(types):
            terminate_flag_seq = cls.create_default_terminate_flag_seq(len(sequence_list[0]))
            sequence_list.append(terminate_flag_seq)
            types.append(TerminateFlag)

        n_type = len(set(types))
        all_different_type = n_type == len(sequence_list)
        assert all_different_type, "all sequences must have different type"

        sequence_dict = {seq.elem_type: seq for seq in sequence_list}

        # final data validation
        cls.validate_sequence_dict(sequence_dict, check_terminate_flag=check_terminate_flags)

        return cls(sequence_dict, metadata, timestamp_seq)  # type: ignore

    @classmethod
    def from_edict_list(
        cls,
        edict_list: List[ElementDict],
        timestamp_seq: Optional[TimeStampSequence] = None,
        metadata: Optional[MetaData] = None,
        check_terminate_flag: bool = True,
    ) -> "EpisodeData":
        r"""Create episode from list of ElementDict
        Args:
            edict_list: list of ElementDict to construct the episode
            timestamp_seq: optional timestamp information
            metadata: optional metadata
            check_terminate_flag: if True, validation of TerminateFlag sequence will be  conducted. (see: validate_terminate_flags function)
        """

        # all edict must have the same keys
        key_set_ref = set(edict_list[0].keys())
        for edict in edict_list:
            assert_equal_with_message(set(edict.keys()), key_set_ref, "key set")

        elem_seq_list = []
        for key in key_set_ref:
            elem_list = []
            for edict in edict_list:
                elem_list.append(edict[key])
            elem_seq = ElementSequence(elem_list)
            elem_seq_list.append(elem_seq)
        return cls.from_seq_list(
            elem_seq_list, timestamp_seq, metadata, check_terminate_flags=check_terminate_flag
        )

    def get_sequence_by_type(self, elem_type: Type[ElementT]) -> ElementSequence[ElementT]:

        if issubclass(elem_type, PrimitiveElementBase):
            return self.sequence_dict[elem_type]  # type: ignore
        elif issubclass(elem_type, CompositeImageBase):
            if elem_type in self.sequence_dict:
                return self.sequence_dict[elem_type]  # type: ignore
            else:
                seqs = [self.sequence_dict[t] for t in elem_type.image_types]
                return create_composite_image_sequence(elem_type, seqs)  # type: ignore
        else:
            assert False, "element with type {} not found".format(elem_type)

    def set_sequence(
        self, elem_type: Type[PrimitiveElementT], seq: ElementSequence[PrimitiveElementT]
    ) -> None:
        """set element sequence corresponding to elem_type"""
        assert issubclass(elem_type, PrimitiveElementBase)
        assert seq.elem_type == elem_type
        self.sequence_dict[elem_type] = seq  # type: ignore
        self.validate_sequence_dict(self.sequence_dict)

    @overload
    def __getitem__(self, index: int) -> ElementDict:
        pass

    @overload
    def __getitem__(self, indices: List[int]) -> "EpisodeData":
        """remove TerminateFlag"""

    @overload
    def __getitem__(self, slicee: slice) -> "EpisodeData":
        """remove TerminateFlag"""

    def __getitem__(self, index_like):
        if isinstance(index_like, int):
            elems = [seq[index_like] for seq in self.sequence_dict.values()]
            return ElementDict(elems)
        elif isinstance(index_like, slice) or isinstance(index_like, list):

            if self.time_stamp_seq is None:
                partial_ts_seq = None
            else:
                partial_ts_seq = TimeStampSequence(self.time_stamp_seq[index_like])

            partial_seq_list = []
            for seq in self.sequence_dict.values():
                if seq.elem_type != TerminateFlag:
                    # TODO(HiroIshida): remove type ignore
                    partial_seq_list.append(ElementSequence(seq[index_like]))  # type: ignore
            return EpisodeData.from_seq_list(
                partial_seq_list, timestamp_seq=partial_ts_seq, metadata=self.metadata
            )
        else:
            assert False

    def slice_by_time(
        self, t_start_data: float, t_end_data: float, t_end_task: Optional[float] = None
    ) -> "EpisodeData":
        """slice episode using times.
        t_start_data: start time of data
        t_end_data: end time of this episode
        t_end_task: input this if t_end_task does not match t_end_data. This is important if TerminateFlag before the terminal index.

        These three values must satisfy following realtions t_start_data < t_end_task < t_end_data

        """
        assert t_start_data < t_end_data

        assert self.time_stamp_seq is not None
        i_start_data = self.time_stamp_seq.index_geq(t_start_data)
        i_end_data = self.time_stamp_seq.index_geq(t_end_data)

        assert i_start_data < i_end_data
        partial_episode = self[i_start_data : i_end_data + 1]

        if t_end_task is None:
            return partial_episode
        else:
            assert t_start_data < t_end_task < t_end_data
            # create flag sequence considering t_end_task
            i_end_task = self.time_stamp_seq.index_geq(t_end_task)
            flag_list_false = [
                TerminateFlag.from_bool(False) for _ in range(i_end_task - i_start_data)
            ]
            flag_list_true = [
                TerminateFlag.from_bool(True) for _ in range(i_end_data - i_end_task + 1)
            ]
            flag_seq = ElementSequence[TerminateFlag](flag_list_false + flag_list_true)

            # bit dirty...
            assert partial_episode.time_stamp_seq is not None  # just for mypy
            time_seq = partial_episode.time_stamp_seq
            seq_dict = copy.deepcopy(partial_episode.sequence_dict)
            # TODO(HiroIshida) remove type-ignore (generic dict..?)
            seq_dict[TerminateFlag] = flag_seq  # type: ignore
            return EpisodeData.from_seq_list(
                list(seq_dict.values()), timestamp_seq=time_seq, metadata=self.metadata
            )

    def save_debug_gif(self, file_path: Union[Path, str], fps: int = 20):
        t: Type[ImageBase]
        if RGBDImage in self.types():
            t = RGBDImage
        elif RGBImage in self.types():
            t = RGBImage
        else:
            assert False, "Currently only RGB or RBGD is supported"

        seq = self.get_sequence_by_type(t)

        clip = ImageSequenceClip([e.to_rgb().numpy() for e in seq], fps=fps)
        clip.write_gif(str(file_path), fps=fps)

    def dump(self, episode_dir_path: pathlib.Path, compress: bool = False) -> None:
        episode_dir_path = episode_dir_path.expanduser()
        episode_dir_path.mkdir(exist_ok=True)

        for elem_type, seq in self.sequence_dict.items():
            seq.dump(episode_dir_path, compress=compress)

        if self.time_stamp_seq is not None:
            self.time_stamp_seq.dump(episode_dir_path)

        with open(episode_dir_path / "metadata.json", mode="w") as f:
            json.dump(self.metadata, f)

    @classmethod
    def load(cls, episode_dir_path: pathlib.Path) -> "EpisodeData":
        episode_dir_path = episode_dir_path.expanduser()
        type_seq_table = ElementSequence.load_all(episode_dir_path)
        time_stamp_seq = TimeStampSequence.load(episode_dir_path)

        episode_dir_path / "metadata.json"
        with open(episode_dir_path / "metadata.json", mode="r") as f:
            metadata = MetaData(json.load(f))
        return cls(type_seq_table, metadata, time_stamp_seq)


@dataclass(frozen=True)
class BundleSpec(HasTypeShapeTable):
    n_episode: int
    n_untouch_episode: int
    n_average: int
    type_shape_table: Dict[Type[ElementBase], Tuple[int, ...]]
    meta_data: MetaData

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["type_shape_table"] = {k.__name__: list(v) for k, v in d["type_shape_table"].items()}

        meta_data_dict = {k: v for (k, v) in self.meta_data.items()}
        d["meta_data"] = meta_data_dict
        return d

    @classmethod
    def from_dict(cls, d: Dict) -> "BundleSpec":
        d["type_shape_table"] = {
            get_element_type(k): tuple(v) for k, v in d["type_shape_table"].items()
        }
        d["meta_data"] = MetaData(d["meta_data"])
        return cls(**d)

    @classmethod
    def load(cls, project_path: Path, postfix: Optional[str] = None) -> "BundleSpec":
        file_path = cls.file_path(project_path, postfix)
        with file_path.open(mode="r") as f:
            d = yaml.safe_load(f)
        return cls.from_dict(d)

    @staticmethod
    def file_path(project_path: Path, postfix: Optional[str] = None) -> Path:
        if postfix is None:
            return project_path / "bundle_spec.yaml"
        else:
            return project_path / "bundle_spec-{}.yaml".format(postfix)

    def dump(self, project_path: Path, postfix: Optional[str] = None) -> None:
        file_path = self.file_path(project_path, postfix)
        with file_path.open(mode="w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)


# If classmethod + lru_cache combination is easy, we should replace this by lru_cache
# If you are unsatisfied by this, please make a PR using lru_cache
# there is something like this https://gist.github.com/pavelpy/69f8cdad94aaf59abcf879bda66bfd1a
_bundle_cache: Dict[Tuple[Path, Optional[str]], "EpisodeBundle"] = {}  # used EpisodeBundle.load


@dataclass
class EpisodeBundle(HasAList[EpisodeData], HasTypeShapeTable):
    """Bundle of episode
    The collection of episodes.
    we call it 'bundle' because 'Dataset' is already used by pytorch
    """

    _episode_list: List[EpisodeData]
    _untouch_episode_list: List[EpisodeData]
    metadata: MetaData
    postfix: Optional[str] = None

    def _get_has_a_list(self) -> List[EpisodeData]:
        return self._episode_list

    @property
    def type_shape_table(self) -> Dict[Type[ElementBase], Tuple[int, ...]]:
        return self._episode_list[0].type_shape_table

    @property
    def spec(self) -> BundleSpec:
        n_average = int(sum([len(data) for data in self._episode_list]) / len(self._episode_list))
        spec = BundleSpec(
            len(self._episode_list),
            len(self._untouch_episode_list),
            n_average,
            self.type_shape_table,
            self.metadata,
        )
        return spec

    @classmethod
    def from_data_list(cls, *args, **kwargs) -> "EpisodeBundle":
        """same as from_episodes. just for backward compatibility"""
        return cls.from_episodes(*args, **kwargs)

    @classmethod
    def from_episodes(
        cls,
        episode_list: List[EpisodeData],
        meta_data: Optional[MetaData] = None,
        shuffle: bool = True,
        n_untouch_episode: int = 5,
        check_duplication: bool = True,
    ) -> "EpisodeBundle":

        # check if episode data dupliation
        if check_duplication:
            hash_list = [e.hash_value for e in episode_list]
            hash_set = set(hash_list)

            n_list = len(hash_list)
            n_set = len(hash_set)
            assert (
                n_list == n_set
            ), "episode duplication found. list length {}, set length {}".format(n_list, n_set)

        if meta_data is None:
            meta_data = MetaData({})

        set_types = set(
            functools.reduce(operator.add, [list(data.types()) for data in episode_list])
        )

        n_type_appeared = len(set_types)
        n_type_expected = len(episode_list[0].types())
        assert_equal_with_message(n_type_appeared, n_type_expected, "num of element type in bundle")

        untouch_episode_list = []
        if n_untouch_episode > 0:
            interval = len(episode_list) // n_untouch_episode
            indices_untouch = [interval * i for i in range(n_untouch_episode)]
            # sorted is necessary because pop changes index
            for idx in sorted(indices_untouch, reverse=True):
                untouch_episode_list.append(episode_list.pop(idx))

        # use fixed random seed where it's scope is only in this function
        # The reason why I used fixed seed is to keep two different shuffled
        # bundle generated has the contents in the same order.
        # This is particulary important say you make a bundl with 5hz and another
        # with 20hz, and do some computation using both of bundle.
        rn = random.Random()
        rn.seed(0)
        if shuffle:
            rn.shuffle(episode_list)

        return cls(episode_list, untouch_episode_list, meta_data)

    @classmethod
    def _load(cls, bundle_dir_path: pathlib.Path, postfix: Optional[str]) -> "EpisodeBundle":
        def load_episodes(str_startswitdth: str):
            episode_names: List[str] = natsort.natsorted(
                [p.name for p in bundle_dir_path.iterdir() if p.name.startswith(str_startswitdth)],
            )  # type: ignore
            episode_list = []
            for episode_name in episode_names:
                episode_dir_path = bundle_dir_path / episode_name
                episode_list.append(EpisodeData.load(episode_dir_path))
            return episode_list

        episode_list = load_episodes("episode")
        untouch_episode_list = load_episodes("untouch_episode")

        metadata_file_path = bundle_dir_path / "metadata.json"
        with metadata_file_path.open(mode="r") as f:
            metadata = MetaData(json.load(f))
        bundle = EpisodeBundle(episode_list, untouch_episode_list, metadata, postfix)
        return bundle

    @classmethod
    def load(cls, project_path: Path, postfix: Optional[str] = None) -> "EpisodeBundle":

        if (project_path, postfix) not in _bundle_cache:

            bundle_file_without_ext = "EpisodeBundle"
            if postfix is not None:
                bundle_file_without_ext += "-{}".format(postfix)

            bundle_tar = bundle_file_without_ext + ".tar"
            bundle_tar_path = project_path / bundle_tar
            if not bundle_tar_path.exists():
                raise FileNotFoundError("budle {} is not found".format(bundle_tar_path))

            with tempfile.TemporaryDirectory() as tmp_dir:
                tmp_dir_path = pathlib.Path(tmp_dir)
                # TODO: use python's tarfile library
                subprocess.run(
                    "cd {} && tar -xf {}".format(tmp_dir_path, bundle_tar_path), shell=True
                )
                bundle_dir_path = tmp_dir_path / bundle_file_without_ext
                bundle = cls._load(bundle_dir_path, postfix)

            # because loading time of bundle is not negligible, we will cache the bundle with thep project_path and postfix key
            # and use it when the query with the same key is asked.
            _bundle_cache[(project_path, postfix)] = bundle

        bundle = _bundle_cache[(project_path, postfix)]
        return bundle

    def dump(
        self,
        project_path: Path,
        postfix: Optional[str] = None,
        exist_ok: bool = False,
        compress: bool = False,
    ) -> None:
        """dump the bundle

        NOTE: tar is great because it's immutable and no trouble when downloading from gdrive
        and it can be easily viewed on common file viewer.

        TODO: Adding option for compression rate control is a future option.
        Please send me a PR. Currently, no compreesion is applied.
        """

        self.postfix = postfix
        bundle_file_without_ext = "EpisodeBundle" + (
            "" if postfix is None else "-{}".format(postfix)
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_dir_path = pathlib.Path(tmp_dir)

            bundle_dir_path = tmp_dir_path / bundle_file_without_ext
            bundle_dir_path.mkdir(parents=True, exist_ok=True)

            for i, episode in enumerate(self._episode_list):
                episode_dir_path = bundle_dir_path / "episode{}".format(i)
                episode.dump(episode_dir_path, compress=compress)

            for i, episode in enumerate(self._untouch_episode_list):
                episode_dir_path = bundle_dir_path / "untouch_episode{}".format(i)
                episode.dump(episode_dir_path, compress=compress)

            metadata_file_path = bundle_dir_path / "metadata.json"
            with metadata_file_path.open(mode="w") as f:
                json.dump(self.metadata, f)

            tarfile = bundle_file_without_ext + ".tar"
            tarfile_path = project_path / tarfile
            if tarfile_path.exists():
                if exist_ok:
                    os.remove(tarfile_path)
                else:
                    raise FileExistsError(
                        "Bundle file {} already exists. remove this file or set exist_ok=True.".format(
                            tarfile_path
                        )
                    )

            # TODO: using python tarfile is clean appoach. If get annoyed, please send a PR
            cmd = "cd {} && tar cf {} *".format(tmp_dir_path, tarfile_path)
            subprocess.run(cmd, shell=True)

        # extra dump just for debugging (the following info is not requried to load bundle)
        self.spec.dump(project_path, postfix)

    def get_untouch_bundle(self) -> "EpisodeBundle":
        """get episode bundle which is not used for training."""
        return EpisodeBundle(self._untouch_episode_list, [], self.metadata, self.postfix)

    def get_touch_bundle(self) -> "EpisodeBundle":
        """get episode bundle which is used for training"""
        return EpisodeBundle(self._episode_list, [], self.metadata, self.postfix)

    def __add__(self, other: "EpisodeBundle") -> "EpisodeBundle":
        """merge two episode bundles
        two bundles must has same type_shape_table, i.e. each bundle have to have the same type
        of element sequence and at the same time the shape/dimension of the each element must
        be the same.
        """
        assert self.type_shape_table == other.type_shape_table
        episode_list_new = self._episode_list + other._episode_list
        untouch_episode_list_new = self._untouch_episode_list + other._untouch_episode_list
        return EpisodeBundle(episode_list_new, untouch_episode_list_new, MetaData({}), None)

    def plot_vector_histories(
        self,
        elem_type: Type[VectorBase],
        project_path: Path,
        hz: Optional[float] = None,
        postfix: Optional[str] = None,
    ) -> None:

        n_vec_dim = self.spec.type_shape_table[elem_type][0]

        fig = plt.figure()
        fig, axs = plt.subplots(n_vec_dim)
        for i_dim, ax in enumerate(axs):

            y_min, y_max = +np.inf, -np.inf  # will be updated in the following loop

            for data in self._episode_list:
                seq = data.get_sequence_by_type(AngleVector)
                single_seq = np.array([e.numpy()[i_dim] for e in seq])
                y_min = min(y_min, np.min(single_seq))
                y_max = max(y_max, np.max(single_seq))

                if hz is None:
                    axs[i_dim].plot(single_seq, color="red", lw=0.5)
                else:
                    time_seq = [i / hz for i in range(len(single_seq))]
                    axs[i_dim].plot(time_seq, single_seq, color="red", lw=0.5)

            margin = 0.2
            diff = y_max - y_min
            axs[i_dim].set_ylim([y_min - diff * margin, y_max + diff * margin])

        for ax in axs:
            ax.grid()

        if hz is None:
            axs[-1].set_xlabel("data point number [-]")
        else:
            axs[-1].set_xlabel("time [s]")

        if postfix is None:
            file_path = project_path / "seq-{0}.png".format(elem_type.__name__)
        else:
            file_path = project_path / "seq-{0}-{1}.png".format(postfix, elem_type.__name__)
        file_name = str(file_path)
        fig.savefig(file_name, format="png", dpi=300)
        print("saved to {}".format(file_name))
