import pytest
from typing import Type

import numpy as np

from mohou.types import AngleVector, RGBDImage, RGBImage, DepthImage, VectorBase, PrimitiveImageBase
from mohou.types import ElementDict
from mohou.types import ElementSequence
from mohou.types import EpisodeData
from mohou.types import MultiEpisodeChunk


def test_elements():

    with pytest.raises(Exception):
        VectorBase(np.zeros(3))
    with pytest.raises(Exception):
        PrimitiveImageBase(np.zeros((3, 3)))


def test_rdb_image_creation():

    with pytest.raises(AssertionError):
        RGBImage(np.random.randint(0, 255, (100, 100), dtype=np.uint8))
    with pytest.raises(AssertionError):
        RGBImage(np.random.randn(100, 100, 3))

    RGBImage(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))


def test_depth_image_creation():

    with pytest.raises(AssertionError):
        DepthImage(np.random.randn(100, 100))
    with pytest.raises(AssertionError):
        DepthImage(np.random.randn(100, 100, 2))
    with pytest.raises(AssertionError):
        DepthImage(np.random.randint(0, 255, (100, 100, 1)))

    DepthImage(np.random.randn(100, 100, 1))


@pytest.mark.parametrize('T', [RGBImage, DepthImage])
def test_images(T: Type[PrimitiveImageBase]):
    img = T.dummy_from_shape((100, 100))
    img2 = T.from_tensor(img.to_tensor())
    np.testing.assert_almost_equal(img._data, img2._data, decimal=5)

    with pytest.raises(AssertionError):
        np.testing.assert_almost_equal(img.randomize()._data, img.randomize()._data, decimal=5)

    img.to_rgb()


def test_rdbd_image():
    rgbd = RGBDImage.dummy_from_shape((100, 100))
    tensor = rgbd.to_tensor()
    assert list(tensor.shape) == [4, 100, 100]

    rgbd.to_rgb()

    rgbd2 = RGBDImage.from_tensor(rgbd.to_tensor())
    for im1, im2 in zip(rgbd.images, rgbd2.images):
        np.testing.assert_almost_equal(im1._data, im2._data, decimal=5)


def test_element_dict():
    rgb = RGBImage.dummy_from_shape((100, 100))
    depth = DepthImage.dummy_from_shape((100, 100))
    dic = ElementDict([rgb, depth])

    assert isinstance(dic[RGBImage], RGBImage)
    assert isinstance(dic[RGBDImage], RGBDImage)

    rgbd = RGBDImage.dummy_from_shape((100, 100))
    dic = ElementDict([rgbd])
    assert isinstance(dic[RGBDImage], RGBDImage)


def test_episode_data_creation():
    image_seq = ElementSequence([RGBImage.dummy_from_shape((100, 100)) for _ in range(10)])
    av_seq = ElementSequence([AngleVector(np.zeros(10)) for _ in range(10)])
    data = EpisodeData((image_seq, av_seq))

    assert set(data.type_shape_table.keys()) == set([AngleVector, RGBImage])


def test_episode_data_assertion_different_size():
    image_seq = ElementSequence([RGBImage.dummy_from_shape((100, 100)) for _ in range(3)])
    av_seq = ElementSequence([AngleVector(np.zeros(10)) for _ in range(10)])

    with pytest.raises(AssertionError):
        EpisodeData((image_seq, av_seq))


def test_episode_data_assertion_type_inconsitency():
    image_seq = ElementSequence([RGBImage.dummy_from_shape((100, 100)) for _ in range(10)])

    with pytest.raises(AssertionError):
        EpisodeData((image_seq, image_seq))


@pytest.fixture(scope='session')
def image_av_chunk():
    def create_sedata():
        image_seq = ElementSequence([RGBImage.dummy_from_shape((100, 100)) for _ in range(10)])
        av_seq = ElementSequence([AngleVector(np.zeros(10)) for _ in range(10)])
        data = EpisodeData((image_seq, av_seq))
        return data
    chunk = MultiEpisodeChunk([create_sedata() for _ in range(100)])
    return chunk


def test_multi_episode_chunk_creation(image_av_chunk):
    chunk = image_av_chunk
    assert set(chunk.type_shape_table.keys()) == set([AngleVector, RGBImage])


def test_multi_episode_chunk_assertion_type_inconsitency():
    image_seq = ElementSequence([RGBImage.dummy_from_shape((100, 100)) for _ in range(10)])
    av_seq = ElementSequence([AngleVector(np.zeros(10)) for _ in range(10)])
    depth_seq = ElementSequence([DepthImage(np.zeros((100, 100, 1))) for _ in range(10)])

    data1 = EpisodeData((image_seq, av_seq))
    data2 = EpisodeData((depth_seq, av_seq))

    with pytest.raises(AssertionError):
        MultiEpisodeChunk([data1, data2])
