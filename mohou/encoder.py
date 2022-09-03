import base64
import pickle
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Generic, List, Optional, Tuple, Type, TypeVar

import numpy as np
import torch
from sklearn.decomposition import PCA

from mohou.model.autoencoder import AutoEncoderBase
from mohou.model.common import ModelBase
from mohou.types import ElementT, EpisodeBundle, ImageT, VectorT, get_element_type
from mohou.utils import assert_equal_with_message, assert_isinstance_with_message

EncoderT = TypeVar("EncoderT", bound="EncoderBase")


@dataclass  # type: ignore
class EncoderBase(ABC, Generic[ElementT]):
    elem_type: Type[ElementT]
    input_shape: Tuple[int, ...]
    output_size: int

    def to_dict(self) -> Dict:
        d: Dict = {}
        d["elem_type"] = self.elem_type.__name__
        d["input_shape"] = self.input_shape
        d["output_size"] = self.output_size
        # TODO: automate
        return d

    @classmethod
    def from_dict(cls: Type[EncoderT], d: Dict) -> EncoderT:
        d["elem_type"] = get_element_type(d["elem_type"])
        d["input_shape"] = tuple(d["input_shape"])
        return cls(**d)

    def forward(self, inp: ElementT, check_size: bool = True) -> np.ndarray:
        assert_isinstance_with_message(inp, self.elem_type)

        if check_size:
            assert_equal_with_message(inp.shape, self.input_shape, "input shape")

        out = self._forward_impl(inp)

        if check_size:
            assert_equal_with_message(out.shape, (self.output_size,), "output shape")

        return out

    def backward(self, inp: np.ndarray, check_size: bool = True) -> ElementT:
        if check_size:
            assert_equal_with_message(inp.shape, (self.output_size,), "input shape")

        out = self._backward_impl(inp)

        if check_size:
            assert_equal_with_message(out.shape, self.input_shape, "input shape")

        return out

    @abstractmethod
    def _forward_impl(self, inp: ElementT) -> np.ndarray:
        pass

    @abstractmethod
    def _backward_impl(self, inp: np.ndarray) -> ElementT:
        pass

    def __eq__(self, other: object) -> bool:
        # DO NOT USE dataclass's default __eq__
        if not isinstance(other, EncoderBase):
            return NotImplemented
        assert type(self) == type(other)
        for attr in ["elem_type", "input_shape", "output_size"]:
            if not self.__dict__[attr] == other.__dict__[attr]:
                return False
        return True


class HasAModel(ABC):
    @abstractmethod
    def _get_model(self) -> ModelBase:
        pass

    def get_device(self) -> Optional[torch.device]:
        return self._get_model().device

    def set_device(self, device: torch.device) -> None:
        self._get_model().put_on_device(device)


@dataclass(eq=False)
class ImageEncoder(EncoderBase[ImageT], HasAModel):
    input_shape: Tuple[int, int, int]
    model: AutoEncoderBase

    def __post_init__(self):
        inp_dummy = self.elem_type.dummy_from_shape(self.input_shape[:2])
        out_dummy = self._forward_impl(inp_dummy)
        assert_equal_with_message(out_dummy.shape, (self.output_size,), "shape")

        inp_reconstucted = self._backward_impl(out_dummy)
        assert_isinstance_with_message(inp_reconstucted, self.elem_type)

    @classmethod
    def from_auto_encoder(cls, model: AutoEncoderBase) -> "ImageEncoder":
        image_type = model.image_type
        np_image_shape = (model.config.n_pixel, model.config.n_pixel, model.channel())
        return cls(image_type, np_image_shape, model.config.n_bottleneck, model)

    def to_dict(self) -> Dict:
        d = super().to_dict()
        d["model"] = base64.b64encode(pickle.dumps(self.model)).decode("utf-8")
        return d

    @classmethod
    def from_dict(cls, d: Dict) -> "ImageEncoder":
        d["model"] = pickle.loads(base64.b64decode(d["model"].encode()))
        return super().from_dict(d)

    def _forward_impl(self, inp: ImageT) -> np.ndarray:
        inp_tensor = inp.to_tensor().unsqueeze(dim=0)
        out_tensor = self.model.encode(inp_tensor).squeeze(dim=0)
        out_numpy = out_tensor.cpu().detach().numpy()
        return out_numpy

    def _backward_impl(self, inp: np.ndarray) -> ImageT:
        inp_tensor = torch.from_numpy(inp).unsqueeze(dim=0).float()
        out_tensor = self.model.decode(inp_tensor).squeeze(dim=0)
        out: ImageT = self.elem_type.from_tensor(out_tensor)
        return out

    def __eq__(self, other: object) -> bool:
        # DO NOT USE dataclass's default __eq__
        if not isinstance(other, ImageEncoder):
            return NotImplemented
        assert type(self) == type(other)

        if not super().__eq__(other):
            return False

        # https://discuss.pytorch.org/t/check-if-models-have-same-weights/4351/2
        model1 = self.model
        model2 = other.model
        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            if p1.data.ne(p2.data).sum() > 0:
                return False
        return True

    def _get_model(self) -> ModelBase:
        return self.model


@dataclass
class VectorIdenticalEncoder(EncoderBase[VectorT]):
    input_shape: Tuple[int]

    @classmethod
    def create(cls, vector_type: Type[VectorT], dimension: int) -> "VectorIdenticalEncoder":
        return cls(vector_type, (dimension,), dimension)

    def _forward_impl(self, inp: VectorT) -> np.ndarray:
        return inp.numpy()

    def _backward_impl(self, inp: np.ndarray) -> VectorT:
        return self.elem_type(inp)


class VectorPCAEncoder(EncoderBase[VectorT]):
    input_shape: Tuple[int]
    pca: PCA

    def __init__(self, vector_type: Type[VectorT], pca: PCA):
        input_shape = (pca.n_features_,)
        output_dimension = pca.n_components
        super().__init__(vector_type, input_shape, output_dimension)

        self.pca = pca

    def to_dict(self) -> Dict:  # type: ignore
        assert False, "currently not supported"

    @classmethod
    def from_dict(cls, d: Dict) -> "ImageEncoder":  # type: ignore
        assert False, "currently not supported"

    def _forward_impl(self, inp: VectorT) -> np.ndarray:
        inp_as_2d = np.expand_dims(inp.numpy(), axis=0)
        out = self.pca.transform(inp_as_2d)
        return out.flatten()

    def _backward_impl(self, inp: np.ndarray) -> VectorT:
        out = self.pca.inverse_transform(np.expand_dims(inp, axis=0))
        return self.elem_type(out.flatten())

    @classmethod
    def from_bundle(
        cls, bundle: EpisodeBundle, vector_type: Type[VectorT], n_out: int
    ) -> "VectorPCAEncoder[VectorT]":
        elem_list: List[VectorT] = []
        for episode in bundle.get_touch_bundle():
            elem_seq = episode.get_sequence_by_type(vector_type)
            elem_list.extend(elem_seq.elem_list)
        mat = np.array([e.numpy() for e in elem_list])
        pca = PCA(n_components=n_out)
        pca.fit(mat)
        return cls(vector_type, pca)
