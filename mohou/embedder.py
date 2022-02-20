from abc import abstractmethod
from typing import Callable, Generic, Optional, Tuple

import numpy as np
import torch
import torchvision

from mohou.types import ElementT, ImageBase, VectorBase


class Embedder(Generic[ElementT]):
    input_shape: Tuple[int, ...]
    output_size: int

    def forward(self, inp: ElementT, check_size: bool = True) -> np.ndarray:
        if check_size:
            assert inp.shape == self.input_shape

        out = self._forward_impl(inp)

        if check_size:
            assert out.shape == (self.output_size,)

        return out

    @abstractmethod
    def _forward_impl(self, inp: ElementT) -> np.ndarray:
        pass


class ImageEmbedder(Embedder[ImageBase]):
    input_shape: Tuple[int, int, int]
    func: Optional[Callable[[torch.Tensor], torch.Tensor]]
    # https://stackoverflow.com/questions/51811024/mypy-type-checking-on-callable-thinks-that-member-variable-is-a-method

    def __init__(
            self,
            func: Callable[[torch.Tensor], torch.Tensor],
            input_shape: Tuple[int, int, int],
            output_size: int,
            check_size: bool = True
    ):

        self.func = func
        self.input_shape = input_shape
        self.output_size = output_size

        if check_size:
            inp_dummy = np.zeros(input_shape)
            out_dummy = self._forward_impl(inp_dummy)  # type: ignore
            assert out_dummy.shape == (output_size,)

    def _forward_impl(self, inp: ImageBase) -> np.ndarray:
        tf = torchvision.transforms.ToTensor()
        inp_tensor = tf(inp).unsqueeze(dim=0).float()
        assert self.func is not None
        out_tensor = self.func(inp_tensor).squeeze()
        out_numpy = out_tensor.cpu().detach().numpy()
        return out_numpy


class IdenticalEmbedder(Embedder[VectorBase]):
    input_shape: Tuple[int]

    def __init__(self, dimension):
        self.input_shape = (dimension,)
        self.output_size = dimension

    def _forward_impl(self, inp: VectorBase) -> np.ndarray:
        return inp
