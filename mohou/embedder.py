from abc import ABC, abstractmethod
from typing import Callable, Generic, Optional, Tuple, Type

import numpy as np
import torch

from mohou.types import ElementT, ImageT, VectorT
from mohou.utils import assert_with_message, assert_isinstance_with_message


class EmbedderBase(ABC, Generic[ElementT]):
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
            assert_with_message(inp.shape, self.input_shape, 'input shape')

        out = self._forward_impl(inp)

        if check_size:
            assert_with_message(out.shape, (self.output_size,), 'output shape')

        return out

    def backward(self, inp: np.ndarray, check_size: bool = True) -> ElementT:
        if check_size:
            assert_with_message(inp.shape, (self.output_size,), 'input shape')

        out = self._backward_impl(inp)

        if check_size:
            assert_with_message(out.shape, self.input_shape, 'input shape')

        return out

    @abstractmethod
    def _forward_impl(self, inp: ElementT) -> np.ndarray:
        pass

    @abstractmethod
    def _backward_impl(self, inp: np.ndarray) -> ElementT:
        pass


class ImageEmbedder(EmbedderBase[ImageT]):
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
        super().__init__(image_type, input_shape, output_size)
        self.func_forward = func_forward
        self.func_backward = func_backward

        if check_callables:
            inp_dummy = self.elem_type.dummy_from_shape(input_shape[:2])
            out_dummy = self._forward_impl(inp_dummy)
            assert_with_message(out_dummy.shape, (output_size,), 'shape')

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


class IdenticalEmbedder(EmbedderBase[VectorT]):
    input_shape: Tuple[int]

    def __init__(self, vector_type: Type[VectorT], dimension: int):
        super().__init__(vector_type, (dimension,), dimension)

    def _forward_impl(self, inp: VectorT) -> np.ndarray:
        return inp.numpy()

    def _backward_impl(self, inp: np.ndarray) -> VectorT:
        return self.elem_type(inp)
