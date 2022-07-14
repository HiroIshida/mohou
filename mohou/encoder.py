from abc import ABC, abstractmethod
from typing import Callable, Generic, List, Optional, Tuple, Type

import numpy as np
import torch
from sklearn.decomposition import PCA

from mohou.types import ElementT, EpisodeBundle, ImageT, VectorT
from mohou.utils import assert_equal_with_message, assert_isinstance_with_message


class EncoderBase(ABC, Generic[ElementT]):
    elem_type: Type[ElementT]
    input_shape: Tuple[int, ...]
    output_size: int

    def __init__(self, elem_type: Type[ElementT], input_shape: Tuple[int, ...], output_size: int):
        self.elem_type = elem_type
        self.input_shape = input_shape
        self.output_size = output_size

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


class ImageEncoder(EncoderBase[ImageT]):
    input_shape: Tuple[int, int, int]
    func_forward: Optional[Callable[[torch.Tensor], torch.Tensor]]
    func_backward: Optional[Callable[[torch.Tensor], torch.Tensor]]
    # https://stackoverflow.com/questions/51811024/mypy-type-checking-on-callable-thinks-that-member-variable-is-a-method

    def __init__(
        self,
        image_type: Type[ImageT],
        func_forward: Callable[[torch.Tensor], torch.Tensor],
        func_backward: Callable[[torch.Tensor], torch.Tensor],
        input_shape: Tuple[int, int, int],
        output_size: int,
        check_callables: bool = True,
    ):
        super().__init__(image_type, input_shape, output_size)
        self.func_forward = func_forward
        self.func_backward = func_backward

        if check_callables:
            inp_dummy = self.elem_type.dummy_from_shape(input_shape[:2])
            out_dummy = self._forward_impl(inp_dummy)
            assert_equal_with_message(out_dummy.shape, (output_size,), "shape")

            inp_reconstucted = self._backward_impl(out_dummy)
            assert_isinstance_with_message(inp_reconstucted, self.elem_type)

    def _forward_impl(self, inp: ImageT) -> np.ndarray:
        inp_tensor = inp.to_tensor().unsqueeze(dim=0)
        assert self.func_forward is not None
        out_tensor = self.func_forward(inp_tensor).squeeze(dim=0)
        out_numpy = out_tensor.cpu().detach().numpy()
        return out_numpy

    def _backward_impl(self, inp: np.ndarray) -> ImageT:
        inp_tensor = torch.from_numpy(inp).unsqueeze(dim=0).float()
        assert self.func_backward is not None
        out_tensor = self.func_backward(inp_tensor).squeeze(dim=0)
        out: ImageT = self.elem_type.from_tensor(out_tensor)
        return out


class VectorIdenticalEncoder(EncoderBase[VectorT]):
    input_shape: Tuple[int]

    def __init__(self, vector_type: Type[VectorT], dimension: int):
        super().__init__(vector_type, (dimension,), dimension)

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
