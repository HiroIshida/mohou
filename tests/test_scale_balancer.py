import numpy as np
import pytest

from mohou.scale_balancer import ScaleBalancer


def test_scale_balancer():
    arr1 = np.array([[1, 2, 3], [4, 5, 6]]).T
    arr2 = np.array([[1, 3, 5], [7, 9, 11]]).T
    balancer = ScaleBalancer.from_array_list([arr1, arr2])

    # test balancer's contents
    assert balancer.dimension == 2
    np.testing.assert_almost_equal(balancer.means, np.array([0.5 * (2 + 3), 0.5 * (5 + 9)]))
    np.testing.assert_almost_equal(balancer.widths, np.array([0.5 * (2 + 4), 0.5 * (2 + 4)]))

    # test application
    vec = np.random.randn(2)
    vec_again = balancer.inverse_apply(balancer.apply(vec))
    np.testing.assert_almost_equal(vec, vec_again)

    # test invalid input
    with pytest.raises(AssertionError):
        ScaleBalancer(np.zeros(3), np.abs(np.random.randn(4)))

    with pytest.raises(AssertionError):
        widths_including_small_value = np.random.randn(4)
        widths_including_small_value[0] = 1e-13
        ScaleBalancer(np.zeros(3), widths_including_small_value)
