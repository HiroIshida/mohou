import pytest

import numpy as np

from mohou.types import AngleVector, ElementSequence, RGBImage, DepthImage
from mohou.types import SingleEpisodeData
from mohou.types import MultiEpisodeDataChunk


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


def test_multi_episode_data_chunk_creation():
    def create_sedata():
        image_seq = ElementSequence([RGBImage(np.zeros((100, 100, 3))) for _ in range(10)])
        av_seq = ElementSequence([AngleVector(np.zeros(10)) for _ in range(10)])
        data = SingleEpisodeData((image_seq, av_seq))
        return data
    chunk = MultiEpisodeDataChunk([create_sedata() for _ in range(100)])

    assert set(chunk.types) == set([AngleVector, RGBImage])


def test_multi_episode_data_chunk_assertion_type_inconsitency():
    image_seq = ElementSequence([RGBImage(np.zeros((100, 100, 3))) for _ in range(10)])
    av_seq = ElementSequence([AngleVector(np.zeros(10)) for _ in range(10)])
    depth_seq = ElementSequence([DepthImage(np.zeros((100, 100))) for _ in range(10)])

    data1 = SingleEpisodeData((image_seq, av_seq))
    data2 = SingleEpisodeData((depth_seq, av_seq))

    with pytest.raises(AssertionError):
        MultiEpisodeDataChunk([data1, data2])
