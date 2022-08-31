import numpy as np
import torch

from mohou.encoding_rule import (
    CovarianceBasedScaleBalancer,
    EncodingRule,
    EncodingRuleBase,
    ScaleBalancerBase,
)
from mohou.types import AngleVector, ElementBase
from mohou.utils import get_type_from_name, splitting_slices


def test_get_element_type():
    assert get_type_from_name("AngleVector", ElementBase) == AngleVector
    assert (
        get_type_from_name("CovarianceBasedScaleBalancer", ScaleBalancerBase)
        == CovarianceBasedScaleBalancer
    )
    assert get_type_from_name("EncodingRule", EncodingRuleBase) == EncodingRule


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
