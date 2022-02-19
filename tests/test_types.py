import pytest

import numpy as np

from mohou.types import AngleVector, ElementSequence, RGBImage, DepthImage
from mohou.types import EpisodeData
from mohou.types import MultiEpisodeChunk


def test_episode_data_creation():
    image_seq = ElementSequence([RGBImage(np.zeros((100, 100, 3))) for _ in range(10)])
    av_seq = ElementSequence([AngleVector(np.zeros(10)) for _ in range(10)])
    data = EpisodeData((image_seq, av_seq))

    assert set(data.types) == set([AngleVector, RGBImage])


def test_episode_data_assertion_different_size():
    image_seq = ElementSequence([RGBImage(np.zeros((100, 100, 3))) for _ in range(3)])
    av_seq = ElementSequence([AngleVector(np.zeros(10)) for _ in range(10)])

    with pytest.raises(AssertionError):
        EpisodeData((image_seq, av_seq))


def test_episode_data_assertion_type_inconsitency():
    image_seq = ElementSequence([RGBImage(np.zeros((100, 100, 3))) for _ in range(10)])
    depth_seq = ElementSequence([DepthImage(np.zeros((100, 100))) for _ in range(10)])

    with pytest.raises(AssertionError):
        EpisodeData((image_seq, depth_seq))


@pytest.fixture(scope='session')
def image_av_chunk():
    def create_sedata():
        image_seq = ElementSequence([RGBImage(np.zeros((100, 100, 3))) for _ in range(10)])
        av_seq = ElementSequence([AngleVector(np.zeros(10)) for _ in range(10)])
        data = EpisodeData((image_seq, av_seq))
        return data
    chunk = MultiEpisodeChunk([create_sedata() for _ in range(100)])
    return chunk


def test_multi_episode_chunk_creation(image_av_chunk):
    chunk = image_av_chunk
    assert set(chunk.types) == set([AngleVector, RGBImage])


def test_multi_episode_chunk_assertion_type_inconsitency():
    image_seq = ElementSequence([RGBImage(np.zeros((100, 100, 3))) for _ in range(10)])
    av_seq = ElementSequence([AngleVector(np.zeros(10)) for _ in range(10)])
    depth_seq = ElementSequence([DepthImage(np.zeros((100, 100))) for _ in range(10)])

    data1 = EpisodeData((image_seq, av_seq))
    data2 = EpisodeData((depth_seq, av_seq))

    with pytest.raises(AssertionError):
        MultiEpisodeChunk([data1, data2])
