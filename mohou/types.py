from abc import ABC, abstractmethod
import copy
from dataclasses import dataclass, asdict
import functools
import operator
import random
from pathlib import Path
import yaml
from typing import Generic, Optional, List, Tuple, Type, TypeVar, Iterator, Sequence, ClassVar, Dict, Union

import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import cv2
import PIL.Image

from mohou.constant import N_DATA_INTACT
from mohou.file import get_project_path, load_object, dump_object
from mohou.image_randomizer import _f_randomize_rgb_image, _f_randomize_depth_image, _f_randomize_gray_image
from mohou.constant import CONTINUE_FLAG_VALUE, TERMINATE_FLAG_VALUE
from mohou.utils import get_all_concrete_leaftypes
from mohou.utils import split_sequence, canvas_to_ndarray
from mohou.utils import assert_with_message, assert_isinstance_with_message

ElementT = TypeVar('ElementT', bound='ElementBase')
PrimitiveElementT = TypeVar('PrimitiveElementT', bound='PrimitiveElementBase')
PrimitiveImageT = TypeVar('PrimitiveImageT', bound='PrimitiveImageBase')
ColorImageT = TypeVar('ColorImageT', bound='ColorImageBase')
CompositeImageT = TypeVar('CompositeImageT', bound='CompositeImageBase')
ImageT = TypeVar('ImageT', bound='ImageBase')
VectorT = TypeVar('VectorT', bound='VectorBase')


CompositeListElementT = TypeVar('CompositeListElementT')


class HasAList(Sequence, Generic[CompositeListElementT]):

    @abstractmethod
    def _get_has_a_list(self) -> List[CompositeListElementT]:
        pass

    def __iter__(self) -> Iterator[CompositeListElementT]:
        return self._get_has_a_list().__iter__()

    def __getitem__(self, indices_like):  # TODO(HiroIshida) add type hints?
        return self._get_has_a_list()[indices_like]

    def __len__(self) -> int:
        return len(self._get_has_a_list())


class ElementBase(ABC):

    def __new__(cls, *args, **kwargs):
        # instantiationg blocking hack. Different but similar to
        # https://stackoverflow.com/a/7990308/7624196
        assert cls.is_concrete_type(),\
            '{} is an abstract class and thus cannot instantiate'.format(cls.__name__)
        # https://stackoverflow.com/questions/59217884/
        return super(ElementBase, cls).__new__(cls)

    @classmethod
    def is_concrete_type(cls):
        return len(cls.__abstractmethods__) == 0 and len(cls.__subclasses__()) == 0

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


class TerminateFlag(VectorBase):

    @classmethod
    def from_bool(cls, flag: bool) -> 'TerminateFlag':
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
    def to_rgb(self, *args, **kwargs) -> 'RGBImage':
        pass

    @abstractmethod
    def resize(self, shape2d_new: Tuple[int, int]) -> None:
        pass


class PrimitiveImageBase(PrimitiveElementBase, ImageBase):
    _channel: ClassVar[int]

    def __init__(self, data: np.ndarray) -> None:
        super().__init__(data)
        assert_with_message(self._data.ndim, 3, 'image_dim')
        assert_with_message(data.shape[2], self.channel(), 'channel')

    @classmethod
    def channel(cls) -> int:
        return cls._channel


class ColorImageBase(PrimitiveImageBase, Generic[ColorImageT]):

    def __init__(self, data: np.ndarray) -> None:
        super().__init__(data)
        assert_with_message(self._data.dtype.type, np.uint8, 'numpy type')

    def to_tensor(self) -> torch.Tensor:
        return torchvision.transforms.ToTensor()(self._data).float()

    @classmethod
    def dummy_from_shape(cls: Type[ColorImageT], shape2d: Tuple[int, int]) -> ColorImageT:
        shape = (shape2d[0], shape2d[1], cls.channel())
        dummy_array = np.random.randint(0, high=255, size=shape, dtype=np.uint8)
        return cls(dummy_array)


class RGBImage(ColorImageBase['RGBImage']):
    _channel: ClassVar[int] = 3

    @classmethod
    def from_file(cls, filename: str) -> 'RGBImage':
        pil_img = PIL.Image.open(filename).convert('RGB')
        arr = np.array(pil_img)
        return cls(arr)

    @classmethod
    def from_tensor(cls, tensor: torch.Tensor) -> 'RGBImage':
        tf = torchvision.transforms.ToPILImage()
        pil_iamge = tf(tensor)
        return cls(np.array(pil_iamge))

    def randomize(self) -> 'RGBImage':
        assert _f_randomize_rgb_image is not None
        rand_image_arr = _f_randomize_rgb_image(self._data)
        return RGBImage(rand_image_arr)

    def to_rgb(self, *args, **kwargs) -> 'RGBImage':
        return self

    def resize(self, shape2d_new: Tuple[int, int]) -> None:
        self._data = cv2.resize(self._data, shape2d_new, interpolation=cv2.INTER_AREA)


class GrayImage(ColorImageBase['GrayImage']):
    _channel: ClassVar[int] = 1

    @classmethod
    def from_tensor(cls, tensor: torch.Tensor) -> 'GrayImage':
        tensor2d = tensor.squeeze(dim=0)
        tf = torchvision.transforms.ToPILImage()
        pil_iamge = tf(tensor2d)
        return cls(np.expand_dims(np.array(pil_iamge), axis=2))

    def randomize(self) -> 'GrayImage':
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
        rgb: RGBImage,
        laplace_kernel_size: int = 3,
        blur_kernel_size: Optional[Tuple[int, int]] = None) -> GrayImage:

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
        assert_with_message(self._data.dtype.type, [np.float16, np.float32, np.float64], 'numpy type')

    def to_tensor(self) -> torch.Tensor:
        data_cutoff = np.maximum(np.minimum(self._data, self._max_value), self._min_value)
        data_normalized = (data_cutoff - self._min_value) / (self._max_value - self._min_value)
        return torch.from_numpy(data_normalized.transpose((2, 0, 1))).contiguous().float()

    @classmethod
    def from_tensor(cls, tensor: torch.Tensor) -> 'DepthImage':
        data = tensor.detach().clone().numpy().transpose((1, 2, 0))
        data_denormalized = data * (cls._max_value - cls._min_value) + cls._min_value
        return cls(data_denormalized)

    def randomize(self) -> 'DepthImage':
        assert _f_randomize_depth_image is not None
        rand_depth_arr = _f_randomize_depth_image(self._data)
        return DepthImage(rand_depth_arr)

    @classmethod
    def dummy_from_shape(cls, shape2d: Tuple[int, int]) -> 'DepthImage':
        shape = (shape2d[0], shape2d[1], cls.channel())
        dummy_array = np.random.rand(*shape)
        return cls(dummy_array)

    def to_rgb(self, *args, **kwargs) -> 'RGBImage':
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
                assert_with_message(image.shape[:2], image_shape, 'image w-h')

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


class RGBDImage(CompositeImageBase):
    image_types = [RGBImage, DepthImage]

    def to_rgb(self, *args, **kwargs) -> RGBImage:
        for image in self.images:
            if isinstance(image, RGBImage):
                return image
        assert False


class ElementDict(Dict[Type[ElementBase], ElementBase]):

    def __init__(self, elems: Sequence[ElementBase]):
        for elem in elems:
            self[elem.__class__] = elem
        assert_with_message(len(set(self.keys())), len(elems), 'num of element')

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


def get_element_type(type_name: str) -> Type[ElementBase]:
    for t in get_all_concrete_leaftypes(ElementBase):
        if type_name == t.__name__:
            return t
    assert False, 'type {} not found'.format(type_name)


class ElementSequence(HasAList[ElementT], Generic[ElementT]):
    elem_type: Optional[Type[ElementT]] = None
    elem_shape: Optional[Tuple[int, ...]] = None
    elem_list: List[ElementT]

    def __init__(self, elem_list: Optional[List[ElementT]] = None):
        if elem_list is None or len(elem_list) == 0:
            self.elem_list = []
        else:
            assert len(set([type(elem) for elem in elem_list])) == 1
            assert len(set([elem.shape for elem in elem_list])) == 1
            self.elem_list = elem_list
            self.elem_type = type(elem_list[0])
            self.elem_shape = elem_list[0].shape

    def _get_has_a_list(self) -> List[ElementT]:
        return self.elem_list

    def append(self, elem: ElementT):
        if self.elem_type is None:
            self.elem_type = type(elem)
        if self.elem_shape is None:
            self.elem_shape = elem.shape
        assert type(elem) == self.elem_type
        assert elem.shape == self.elem_shape
        self.elem_list.append(elem)

    def get_partial(self, indices: List[int]) -> 'ElementSequence[ElementT]':
        elems = [self.elem_list[idx] for idx in indices]
        return ElementSequence(elems)


def create_composite_image_sequence(
        composite_image_type: Type[CompositeImageT],
        elem_seqs: List[ElementSequence[PrimitiveImageBase]]) -> ElementSequence[CompositeImageT]:

    n_len_seq = len(elem_seqs[0])
    composite_image_seq = ElementSequence[CompositeImageT]([])
    for i in range(n_len_seq):
        composite_image = composite_image_type([seq[i] for seq in elem_seqs])
        composite_image_seq.append(composite_image)
    return composite_image_seq


class TypeShapeTableMixin:

    def types(self) -> List[Type[ElementBase]]:
        return list(self.type_shape_table.keys())  # type: ignore


@dataclass(frozen=True)
class EpisodeData(HasAList[ElementSequence], TypeShapeTableMixin):
    sequence_list: List[ElementSequence]
    type_shape_table: Dict[Type[ElementBase], Tuple[int, ...]]

    def __post_init__(self):
        ef_seq = self.get_sequence_by_type(TerminateFlag)
        self.check_terminate_seq(ef_seq)

    def _get_has_a_list(self) -> List[ElementSequence]:
        return self.sequence_list

    @staticmethod
    def create_default_terminate_flag_seq(n_length) -> ElementSequence[TerminateFlag]:
        flag_lst = [TerminateFlag.from_bool(False) for _ in range(n_length - 1)]
        flag_lst.append(TerminateFlag.from_bool(True))
        elem_seq = ElementSequence(flag_lst)
        return elem_seq

    @staticmethod
    def check_terminate_seq(ef_seq: ElementSequence[TerminateFlag]):
        # first index must be CONTINUE
        assert ef_seq[0].numpy().item() == CONTINUE_FLAG_VALUE
        # last index must be END
        assert ef_seq[-1].numpy().item() == TERMINATE_FLAG_VALUE

        # sequence must be like ffffffftttttt not ffffttffftttt
        change_count = 0
        for i in range(len(ef_seq) - 1):
            if ef_seq[i + 1].numpy().item() != ef_seq[i].numpy().item():
                change_count += 1
        assert change_count == 1

    @classmethod
    def from_seq_list(cls, sequence_list: List[ElementSequence]):

        for sequence in sequence_list:
            assert isinstance(sequence, ElementSequence)

        all_same_length = len(set(map(len, sequence_list))) == 1
        assert all_same_length

        types = [type(seq[0]) for seq in sequence_list]

        if TerminateFlag not in set(types):
            terminate_flag_seq = cls.create_default_terminate_flag_seq(len(sequence_list[0]))
            sequence_list.append(terminate_flag_seq)
            types.append(TerminateFlag)

        shapes = [seq[0].shape for seq in sequence_list]
        type_shape_table = dict({t: s for (t, s) in zip(types, shapes)})

        n_type = len(set(types))
        all_different_type = n_type == len(sequence_list)
        assert all_different_type, 'all sequences must have different type'
        return cls(sequence_list, type_shape_table)

    def get_sequence_by_type(self, elem_type: Type[ElementT]) -> ElementSequence[ElementT]:

        def get_sequence_by_primitive_type(elem_type):
            for seq in self.sequence_list:
                if isinstance(seq[0], elem_type):
                    # thanks to all_different_type
                    return seq
            assert False, 'element with type {} not found'.format(elem_type)

        if issubclass(elem_type, PrimitiveElementBase):
            return get_sequence_by_primitive_type(elem_type)  # type: ignore
        elif issubclass(elem_type, CompositeImageBase):
            seqs = [get_sequence_by_primitive_type(t) for t in elem_type.image_types]
            return create_composite_image_sequence(elem_type, seqs)  # type: ignore
        else:
            assert False, 'element with type {} not found'.format(elem_type)

    def save_debug_gif(self, filename: str, fps: int = 20):
        t: Type[ImageBase]
        if RGBDImage in self.types():
            t = RGBDImage
        elif RGBImage in self.types():
            t = RGBImage
        else:
            assert False, 'Currently only RGB or RBGD is supported'

        seq = self.get_sequence_by_type(t)
        from moviepy.editor import ImageSequenceClip
        clip = ImageSequenceClip([e.to_rgb().numpy() for e in seq], fps=fps)
        clip.write_gif(filename, fps=fps)


ExtraInfoType = Optional[Dict[str, Union[str, int, float]]]


@dataclass(frozen=True)
class ChunkSpec(TypeShapeTableMixin):
    n_episode: int
    n_episode_intact: int
    n_average: int
    type_shape_table: Dict[Type[ElementBase], Tuple[int, ...]]
    extra_info: ExtraInfoType = None

    def to_dict(self) -> Dict:
        d = asdict(self)
        d['type_shape_table'] = {
            k.__name__: list(v)
            for k, v in d['type_shape_table'].items()}
        return d

    @classmethod
    def from_dict(cls, d: Dict) -> 'ChunkSpec':
        d['type_shape_table'] = {
            get_element_type(k): tuple(v)
            for k, v in d['type_shape_table'].items()}
        return cls(**d)


_chunk_cache: Dict[Tuple[str, Optional[str]], 'MultiEpisodeChunk'] = {}  # used MultiEpisodeChunk.load


@dataclass
class MultiEpisodeChunk(HasAList[EpisodeData], TypeShapeTableMixin):
    data_list: List[EpisodeData]
    data_list_intact: List[EpisodeData]
    type_shape_table: Dict[Type[ElementBase], Tuple[int, ...]]
    spec: ChunkSpec
    _postfix: Optional[str] = None

    def _get_has_a_list(self) -> List[EpisodeData]:
        return self.data_list

    @classmethod
    def from_data_list(cls,
                       data_list: List[EpisodeData],
                       extra_info: ExtraInfoType = None,
                       shuffle: bool = True,
                       with_intact_data: bool = True) -> 'MultiEpisodeChunk':

        set_types = set(functools.reduce(
            operator.add,
            [list(data.types()) for data in data_list]))

        n_type_appeared = len(set_types)
        n_type_expected = len(data_list[0].types())
        assert_with_message(n_type_appeared, n_type_expected, 'num of element in chunk')

        data_list_intact = []
        if with_intact_data:
            assert N_DATA_INTACT > 0
            interval = len(data_list) // N_DATA_INTACT
            indices_intact = [interval * i for i in range(N_DATA_INTACT)]
            # sorted is necessary because pop changes index
            for idx in sorted(indices_intact, reverse=True):
                data_list_intact.append(data_list.pop(idx))

        if shuffle:
            random.shuffle(data_list)

        type_shape_table = data_list[0].type_shape_table

        # chunk spec
        n_average = int(sum([len(data[0]) for data in data_list]) / len(data_list))
        spec = ChunkSpec(len(data_list), len(data_list_intact), n_average, type_shape_table, extra_info)
        return cls(data_list, data_list_intact, type_shape_table, spec)

    def get_element_shape(self, elem_type: Type[ElementBase]) -> Tuple[int, ...]:
        return self.type_shape_table[elem_type]

    @classmethod
    def load(cls, project_name: str, postfix: Optional[str] = None) -> 'MultiEpisodeChunk':
        if project_name not in _chunk_cache:
            _chunk_cache[(project_name, postfix)] = load_object(cls, project_name, postfix)
        chunk = _chunk_cache[(project_name, postfix)]
        assert chunk._postfix == postfix
        return chunk

    @staticmethod
    def spec_file_path(project_name: str, postfix: Optional[str] = None) -> Path:
        project_path = get_project_path(project_name)
        if postfix is None:
            yaml_file_path = project_path / 'chunk_spec.yaml'
        else:
            yaml_file_path = project_path / 'chunk_spec-{}.yaml'.format(postfix)
        return yaml_file_path

    @classmethod
    def load_spec(cls, project_name: str, postfix: Optional[str] = None) -> ChunkSpec:
        yaml_file_path = cls.spec_file_path(project_name, postfix)
        with yaml_file_path.open(mode='r') as f:
            d = yaml.safe_load(f)
            spec = ChunkSpec.from_dict(d)
        return spec

    def dump(self, project_name: str, postfix: Optional[str] = None) -> None:
        self._postfix = postfix
        dump_object(self, project_name, postfix)
        yaml_file_path = self.spec_file_path(project_name, postfix)
        with yaml_file_path.open(mode='w') as f:
            yaml.dump(self.spec.to_dict(), f, default_flow_style=False, sort_keys=False)

    def get_intact_chunk(self) -> 'MultiEpisodeChunk':
        return MultiEpisodeChunk(self.data_list_intact, [], self.type_shape_table, self.spec)

    def get_not_intact_chunk(self) -> 'MultiEpisodeChunk':
        return MultiEpisodeChunk(self.data_list, [], self.type_shape_table, self.spec)

    def merge(self, other: 'MultiEpisodeChunk') -> None:
        keys_self = set(self.types())
        keys_other = set(other.types())
        assert keys_other.issubset(keys_self)  # TODO(HiroIshida) current limitation, and easily remove this assertion
        keys_common = keys_self.intersection(keys_other)

        def filter_episode_data_list(episode_data_list: List[EpisodeData]):
            # TODO(HiroIshida) not efficient at all...
            episode_data_list_filtered = []
            for episode_data in episode_data_list:
                seqs = []
                for key in keys_common:
                    seqs.append(episode_data.get_sequence_by_type(key))
                episode_data_list_filtered.append(EpisodeData.from_seq_list(seqs))
            assert len(episode_data_list) == len(episode_data_list_filtered)
            return episode_data_list_filtered

        data_list_new = other.data_list
        data_list_intact_new = other.data_list_intact
        data_list_new.extend(filter_episode_data_list(self.data_list))
        self.data_list = data_list_new
        self.data_list_intact = data_list_intact_new
        self.type_shape_table = other.type_shape_table

    def plot_vector_histories(self, elem_type: Type[VectorBase], project_name: Optional[str] = None) -> None:
        n_vec_dim = self.spec.type_shape_table[elem_type][0]

        fig = plt.figure()
        gs = fig.add_gridspec(n_vec_dim, hspace=0)
        axs = gs.subplots(sharex=True, sharey=True)

        for i_dim, ax in enumerate(axs):
            for data in self.data_list:
                seq = data.get_sequence_by_type(AngleVector)
                single_seq = np.array([e.numpy()[i_dim] for e in seq])
                axs[i_dim].plot(single_seq, color='red', lw=0.5)

        for ax in axs:
            ax.grid()
        if project_name is None:
            plt.show()
        else:
            project_path = get_project_path(project_name)
            file_path = project_path / 'seq-{}.png'.format(elem_type.__name__)
            file_name = str(file_path)
            fig.savefig(file_name, format='png', dpi=300)
            print('saved to {}'.format(file_name))
