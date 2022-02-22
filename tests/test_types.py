import pytest

import numpy as np

from mohou.types import AngleVector, ElementSequence, RGBDImage, RGBImage, DepthImage, VectorBase, SingleImageBase
from mohou.types import EpisodeData
from mohou.types import MultiEpisodeChunk


def test_elements():
    with pytest.raises(Exception):
        VectorBase(np.zeros(3))
    with pytest.raises(Exception):
        SingleImageBase(np.zeros((3, 3)))


def test_rdb_image():
    rgb = RGBImage.dummy_from_shape((100, 100))
    tensor = rgb.to_tensor()
    assert list(tensor.shape) == [3, 100, 100]


def test_depth_image():
    dimage = DepthImage.dummy_from_shape((100, 100))
    tensor = dimage.to_tensor()
    assert list(tensor.shape) == [1, 100, 100]


def test_rdbd_image():
    rgbd = RGBDImage.dummy_from_shape((100, 100))
    tensor = rgbd.to_tensor()
    assert list(tensor.shape) == [4, 100, 100]


def test_episode_data_creation():
    image_seq = ElementSequence([RGBImage(np.zeros((100, 100, 3))) for _ in range(10)])
    av_seq = ElementSequence([AngleVector(np.zeros(10)) for _ in range(10)])
    data = EpisodeData((image_seq, av_seq))

    assert set(data.type_shape_table.keys()) == set([AngleVector, RGBImage])


def test_episode_data_assertion_different_size():
    image_seq = ElementSequence([RGBImage(np.zeros((100, 100, 3))) for _ in range(3)])
    av_seq = ElementSequence([AngleVector(np.zeros(10)) for _ in range(10)])

    with pytest.raises(AssertionError):
        EpisodeData((image_seq, av_seq))


def test_episode_data_assertion_type_inconsitency():
    image_seq = ElementSequence([RGBImage(np.zeros((100, 100, 3))) for _ in range(10)])

    with pytest.raises(AssertionError):
        EpisodeData((image_seq, image_seq))


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
    assert set(chunk.type_shape_table.keys()) == set([AngleVector, RGBImage])


def test_multi_episode_chunk_assertion_type_inconsitency():
    image_seq = ElementSequence([RGBImage(np.zeros((100, 100, 3))) for _ in range(10)])
    av_seq = ElementSequence([AngleVector(np.zeros(10)) for _ in range(10)])
    depth_seq = ElementSequence([DepthImage(np.zeros((100, 100))) for _ in range(10)])

    data1 = EpisodeData((image_seq, av_seq))
    data2 = EpisodeData((depth_seq, av_seq))

    with pytest.raises(AssertionError):
        MultiEpisodeChunk([data1, data2])
