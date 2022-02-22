from abc import ABC, abstractmethod
from typing import Callable, Generic, Optional, Tuple, Type

import numpy as np
import torch

from mohou.types import ElementT, ImageT, VectorT


class Embedder(ABC, Generic[ElementT]):
    elem_type: Type[ElementT]
    input_shape: Tuple[int, ...]
    output_size: int

    def forward(self, inp: ElementT, check_size: bool = True) -> np.ndarray:
        assert isinstance(inp, self.elem_type)

        if check_size:
            assert inp.shape == self.input_shape

        out = self._forward_impl(inp)

        if check_size:
            assert out.shape == (self.output_size,)

        return out

    def backward(self, inp: np.ndarray, check_size: bool = True) -> ElementT:
        if check_size:
            assert inp.shape == (self.output_size,)

        out = self._backward_impl(inp)

        if check_size:
            assert out.shape == self.input_shape

        return out

    @abstractmethod
    def _forward_impl(self, inp: ElementT) -> np.ndarray:
        pass

    @abstractmethod
    def _backward_impl(self, inp: np.ndarray) -> ElementT:
        pass


class ImageEmbedder(Embedder[ImageT]):
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
            check_callables: bool = True
    ):
        self.elem_type = image_type
        self.func_forward = func_forward
        self.func_backward = func_backward
        self.input_shape = input_shape
        self.output_size = output_size

        if check_callables:
            inp_dummy = self.elem_type.dummy_from_shape(input_shape[:2])
            out_dummy = self._forward_impl(inp_dummy)
            assert out_dummy.shape == (output_size,)

            inp_reconstucted = self._backward_impl(out_dummy)
            assert isinstance(inp_reconstucted, self.elem_type)

    def _forward_impl(self, inp: ImageT) -> np.ndarray:
        inp_tensor = inp.to_tensor().unsqueeze(dim=0)
        assert self.func_forward is not None
        out_tensor = self.func_forward(inp_tensor).squeeze()
        out_numpy = out_tensor.cpu().detach().numpy()
        return out_numpy

    def _backward_impl(self, inp: np.ndarray) -> ImageT:
        inp_tensor = torch.from_numpy(inp).unsqueeze(dim=0).float()
        assert self.func_backward is not None
        out_tensor = self.func_backward(inp_tensor).squeeze()
        out: ImageT = self.elem_type.from_tensor(out_tensor)
        return out


class IdenticalEmbedder(Embedder[VectorT]):
    input_shape: Tuple[int]

    def __init__(self, vector_type: Type[VectorT], dimension: int):
        self.elem_type = vector_type
        self.input_shape = (dimension,)
        self.output_size = dimension

    def _forward_impl(self, inp: VectorT) -> np.ndarray:
        return inp

    def _backward_impl(self, inp: np.ndarray) -> VectorT:
        return self.elem_type(inp)
