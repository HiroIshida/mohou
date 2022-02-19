import pytest

import numpy as np

from mohou.types import AngleVector, ElementSequence, RGBImage, DepthImage
from mohou.types import SingleEpisodeData


def test_single_episode_data_creation():
    image_seq = ElementSequence([RGBImage(np.zeros((100, 100, 3))) for _ in range(10)])
    av_seq = ElementSequence([AngleVector(np.zeros(10)) for _ in range(10)])
    data = SingleEpisodeData((image_seq, av_seq))

    assert set(data.types) == set([AngleVector, RGBImage])


def test_single_episode_data_assertion_different_size():
    image_seq = ElementSequence([RGBImage(np.zeros((100, 100, 3))) for _ in range(3)])
    av_seq = ElementSequence([AngleVector(np.zeros(10)) for _ in range(10)])

    with pytest.raises(AssertionError):
        SingleEpisodeData((image_seq, av_seq))


def test_single_episode_data_assertion_type_inconsitency():
    image_seq = ElementSequence([RGBImage(np.zeros((100, 100, 3))) for _ in range(10)])
    depth_seq = ElementSequence([DepthImage(np.zeros((100, 100))) for _ in range(10)])

    with pytest.raises(AssertionError):
        SingleEpisodeData((image_seq, depth_seq))
