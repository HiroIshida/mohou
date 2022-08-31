from typing import Generic

import numpy as np
import torch

from mohou.encoding_rule import EncodingRuleBase, ScaleBalancerBase
from mohou.model.autoencoder import AutoEncoderBase
from mohou.types import ElementBase, ImageT, RGBImage
from mohou.utils import get_all_concrete_leaftypes, get_type_from_name, splitting_slices


class FooBase(Generic[ImageT]):
    pass


class Foo1(FooBase[RGBImage]):
    pass


class Foo2(FooBase[RGBImage]):
    pass


def test_get_all_concrete_leaftypes():
    # this check if this function works for generic classes
    lst = get_all_concrete_leaftypes(FooBase)
    assert set(lst) == {Foo1, Foo2}


def test_get_element_type():
    for base_type in [ElementBase, ScaleBalancerBase, EncodingRuleBase, AutoEncoderBase]:
        for t in get_all_concrete_leaftypes(base_type):
            name = t.__name__
            assert get_type_from_name(name, base_type) == t


def test_splitting_slicers():
    n = 10
    tensor = torch.ones(n, 100, 100)
    array = np.ones((n, 100, 100))

    for obj in (tensor, array):
        for i in range(n):
            obj[i] *= i  # type: ignore

    n_elem_list = [3, 3, 4]
    for obj in (tensor, array):
        obj1, obj2, obj3 = [obj[sl] for sl in splitting_slices(n_elem_list)]  # type: ignore
        assert obj1[0][0][0] == 0
        assert obj1[-1][0][0] == 2

        assert obj2[0][0][0] == 3
        assert obj2[-1][0][0] == 5

        assert obj3[0][0][0] == 6
        assert obj3[-1][0][0] == 9
