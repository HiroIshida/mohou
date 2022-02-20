from abc import abstractmethod
from typing import Callable, Generic, Optional, Tuple, Type

import numpy as np
import torch
import torchvision

from mohou.types import AngleVector, ElementT, ImageT, RGBImage, VectorT


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
            func_forward: Callable[[torch.Tensor], torch.Tensor],
            func_backward: Callable[[torch.Tensor], torch.Tensor],
            input_shape: Tuple[int, int, int],
            output_size: int,
            check_size: bool = True
    ):

        self.func_forward = func_forward
        self.func_backward = func_backward
        self.input_shape = input_shape
        self.output_size = output_size

        if check_size:
            inp_dummy = np.zeros(input_shape)
            out_dummy = self._forward_impl(inp_dummy)  # type: ignore
            assert out_dummy.shape == (output_size,)

    @abstractmethod
    def image_type(self) -> Type[ImageT]:
        pass

    def _forward_impl(self, inp: ImageT) -> np.ndarray:
        tf = torchvision.transforms.ToTensor()
        inp_tensor = tf(inp).unsqueeze(dim=0).float()
        assert self.func_forward is not None
        out_tensor = self.func_forward(inp_tensor).squeeze()
        out_numpy = out_tensor.cpu().detach().numpy()
        return out_numpy

    def _backward_impl(self, inp: np.ndarray) -> ImageT:
        inp_tensor = torch.from_numpy(inp).unsqueeze(dim=0).float()
        assert self.func_backward is not None
        out_tensor = self.func_backward(inp_tensor).squeeze()

        tf = torchvision.transforms.ToPILImage()
        image_type: Type = self.image_type()
        out = image_type(tf(out_tensor))
        return out


class RGBImageEmbedder(ImageEmbedder[RGBImage]):

    def image_type(self) -> Type[RGBImage]:
        return RGBImage


class IdenticalEmbedder(Embedder[VectorT]):
    input_shape: Tuple[int]

    def __init__(self, dimension):
        self.input_shape = (dimension,)
        self.output_size = dimension

    @abstractmethod
    def vector_type(self) -> Type[VectorT]:
        pass

    def _forward_impl(self, inp: VectorT) -> np.ndarray:
        return inp

    def _backward_impl(self, inp: np.ndarray) -> VectorT:
        return self.vector_type()(inp)


class AngleVectorIdenticalEmbedder(IdenticalEmbedder):

    def vector_type(self) -> Type[AngleVector]:
        return AngleVector
