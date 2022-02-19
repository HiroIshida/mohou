from abc import abstractmethod
from typing import Callable, Generic, Optional, Tuple

import numpy as np
import torch
import torchvision

from mohou.types import ElementT, ImageBase


class DimensionReducer(Generic[ElementT]):
    input_shape: Tuple[int, ...]
    output_size: int

    def __call__(self, inp: ElementT, check_size: bool = True) -> np.ndarray:
        if check_size:
            assert inp.shape == self.input_shape

        out = self.reducer_impl(inp)

        if check_size:
            assert out.shape == (self.output_size,)

        return out

    @abstractmethod
    def reducer_impl(self, inp: ElementT) -> np.ndarray:
        pass


class ImageDimensionReducer(DimensionReducer[ImageBase]):
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
            out_dummy = self.reducer_impl(inp_dummy)  # type: ignore
            assert out_dummy.shape == (output_size,)

    def reducer_impl(self, inp: ImageBase) -> np.ndarray:
        tf = torchvision.transforms.ToTensor()
        inp_tensor = tf(torch.from_numpy(inp).float()).unsqueeze(dim=0)
        assert self.func is not None
        out_tensor = self.func(inp_tensor).squeeze()
        out_numpy = out_tensor.cpu().detach().numpy()
        return out_numpy
