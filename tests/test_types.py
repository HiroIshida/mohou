import pytest
from typing import Type
import copy
import pickle

import numpy as np

from mohou.types import VectorBase, AngleVector, RGBDImage, RGBImage, DepthImage, PrimitiveImageBase, TerminateFlag, GripperState
from mohou.types import ElementDict
from mohou.types import ElementSequence
from mohou.types import EpisodeData
from mohou.types import ChunkSpec
from mohou.types import MultiEpisodeChunk

from test_file import tmp_project_name  # noqa


def test_elements():

    with pytest.raises(Exception):
        VectorBase(np.zeros(3))
    with pytest.raises(Exception):
        PrimitiveImageBase(np.zeros((3, 3)))


def test_gripper_state():
    for arr in [np.zeros(2), np.ones(2)]:
        gs = GripperState(arr.astype(bool))
        gs_reconstructed = GripperState.from_tensor(gs.to_tensor())
        np.testing.assert_almost_equal(gs._data, gs_reconstructed._data)


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
def test_rgb_and_depth(T: Type[PrimitiveImageBase]):
    img = T.dummy_from_shape((100, 100))
    img2 = T.from_tensor(img.to_tensor())
    np.testing.assert_almost_equal(img._data, img2._data, decimal=5)

    with pytest.raises(AssertionError):
        np.testing.assert_almost_equal(img.randomize()._data, img.randomize()._data, decimal=5)

    img.to_rgb()
    img.resize((224, 224))
    assert img.shape == (224, 224, img.channel())

    img.resize((30, 30))
    assert img.shape == (30, 30, img.channel())


def test_rdbd_image():
    rgbd = RGBDImage.dummy_from_shape((100, 100))
    tensor = rgbd.to_tensor()
    assert list(tensor.shape) == [4, 100, 100]

    rgbd2 = RGBDImage.from_tensor(rgbd.to_tensor())
    for im1, im2 in zip(rgbd.images, rgbd2.images):
        np.testing.assert_almost_equal(im1._data, im2._data, decimal=5)

    rgbd.to_rgb()

    rgbd.resize((224, 224))
    assert rgbd.shape == (224, 224, rgbd.channel())

    rgbd.resize((30, 30))
    assert rgbd.shape == (30, 30, rgbd.channel())

    rgb = RGBImage.dummy_from_shape((100, 100))
    depth = DepthImage.dummy_from_shape((90, 90))  # inconsistent size

    with pytest.raises(AssertionError):
        RGBDImage([rgb, depth])

    depth = DepthImage.dummy_from_shape((100, 100))  # inconsistent size

    with pytest.raises(AssertionError):  # order mismatch
        RGBDImage([depth, rgb])


def test_element_dict():
    rgb = RGBImage.dummy_from_shape((100, 100))
    depth = DepthImage.dummy_from_shape((100, 100))
    dic = ElementDict([rgb, depth])

    assert isinstance(dic[RGBImage], RGBImage)
    assert isinstance(dic[RGBDImage], RGBDImage)

    rgbd = RGBDImage.dummy_from_shape((100, 100))
    dic = ElementDict([rgbd])
    assert isinstance(dic[RGBDImage], RGBDImage)


def test_element_sequence():
    # start from empty list
    elem_seq = ElementSequence[RGBImage]()
    elem_seq.append(RGBImage.dummy_from_shape((100, 100)))
    assert elem_seq.elem_shape == (100, 100, 3)
    assert elem_seq.elem_type == RGBImage

    with pytest.raises(AssertionError):
        elem_seq.append(RGBImage.dummy_from_shape((100, 101)))  # invalid size

    # start from non empty list
    class TorqueVector(VectorBase):
        pass

    elem_seq = ElementSequence[AngleVector]([AngleVector(np.zeros(3)) for _ in range(10)])
    with pytest.raises(AssertionError):
        elem_seq.append(TorqueVector(np.zeros(3)))  # invalid type


def test_episode_data():
    # creation
    image_seq = ElementSequence([RGBImage.dummy_from_shape((100, 100)) for _ in range(10)])
    av_seq = ElementSequence([AngleVector(np.random.randn(10)) for _ in range(10)])
    data = EpisodeData.from_seq_list([image_seq, av_seq])

    assert set(data.type_shape_table.keys()) == set([AngleVector, RGBImage, TerminateFlag])

    # split
    indices = [0, 2, 4]
    partial_data = data.get_partial(indices)
    image_seq_partial = partial_data.get_sequence_by_type(RGBImage)
    av_seq_partial = partial_data.get_sequence_by_type(AngleVector)

    for i, j in enumerate(indices):
        np.testing.assert_equal(image_seq_partial[i].numpy(), image_seq[j].numpy())
        np.testing.assert_equal(av_seq_partial[i].numpy(), av_seq[j].numpy())

    # split using custom flag_seq
    flag_seq = ElementSequence([TerminateFlag.from_bool(b) for b in [False, False, True]])
    partial_data = data.get_partial(indices, flag_seq)


def test_episode_data_assertion_different_size():
    image_seq = ElementSequence([RGBImage.dummy_from_shape((100, 100)) for _ in range(3)])
    av_seq = ElementSequence([AngleVector(np.zeros(10)) for _ in range(10)])

    with pytest.raises(AssertionError):
        EpisodeData.from_seq_list([image_seq, av_seq])


def test_episode_data_assertion_type_inconsitency():
    image_seq = ElementSequence([RGBImage.dummy_from_shape((100, 100)) for _ in range(10)])

    with pytest.raises(AssertionError):
        EpisodeData.from_seq_list([image_seq, image_seq])


@pytest.fixture(scope='session')
def image_av_chunk():
    def create_edata(n_length):
        image_seq = ElementSequence([RGBImage.dummy_from_shape((100, 100)) for _ in range(n_length)])
        av_seq = ElementSequence([AngleVector(np.zeros(10)) for _ in range(n_length)])
        data = EpisodeData.from_seq_list([image_seq, av_seq])
        return data
    lst = [create_edata(10) for _ in range(20)]
    chunk = MultiEpisodeChunk.from_data_list(lst)
    return chunk


@pytest.fixture(scope='session')
def image_av_chunk_uneven():
    def create_edata(n_length):
        image_seq = ElementSequence([RGBImage.dummy_from_shape((100, 100)) for _ in range(n_length)])
        av_seq = ElementSequence([AngleVector(np.zeros(10)) for _ in range(n_length)])
        data = EpisodeData.from_seq_list([image_seq, av_seq])
        return data
    lst = [create_edata(10) for _ in range(20)]
    lst.append(create_edata(13))
    chunk = MultiEpisodeChunk.from_data_list(lst, shuffle=False, with_intact_data=False)
    return chunk


def test_chunk_spec():
    types = {RGBImage: (100, 100, 3), AngleVector: (7,)}
    spec = ChunkSpec(10, 5, types)
    spec_reconstructed = ChunkSpec.from_dict(spec.to_dict())
    assert pickle.dumps(spec) == pickle.dumps(spec_reconstructed)


def test_multi_episode_chunk(image_av_chunk, tmp_project_name):  # noqa
    chunk: MultiEpisodeChunk = image_av_chunk
    assert set(chunk.type_shape_table.keys()) == set([AngleVector, RGBImage, TerminateFlag])

    chunk.dump(tmp_project_name)
    loaded = chunk.load(tmp_project_name)
    assert pickle.dumps(chunk) == pickle.dumps(loaded)

    chunk_spec = chunk.get_spec()
    chunk_spec_loaded = MultiEpisodeChunk.load_spec(tmp_project_name)
    assert pickle.dumps(chunk_spec) == pickle.dumps(chunk_spec_loaded)


def test_multi_episode_chunk_assertion_type_inconsitency():
    image_seq = ElementSequence([RGBImage.dummy_from_shape((100, 100)) for _ in range(10)])
    av_seq = ElementSequence([AngleVector(np.zeros(10)) for _ in range(10)])
    depth_seq = ElementSequence([DepthImage(np.zeros((100, 100, 1))) for _ in range(10)])

    data1 = EpisodeData.from_seq_list([image_seq, av_seq])
    data2 = EpisodeData.from_seq_list([depth_seq, av_seq])

    with pytest.raises(AssertionError):
        MultiEpisodeChunk.from_data_list([data1, data2])


def test_multi_episode_chunk_merge(image_av_chunk):
    chunk: MultiEpisodeChunk = copy.deepcopy(image_av_chunk)
    chunk.merge(copy.deepcopy(chunk))
    assert len(chunk.data_list) == len(image_av_chunk.data_list) * 2

    # OK
    image_seq = ElementSequence([RGBImage.dummy_from_shape((100, 100)) for _ in range(10)])
    data = EpisodeData.from_seq_list([image_seq])
    chunk2 = MultiEpisodeChunk.from_data_list([data], with_intact_data=False)
    chunk: MultiEpisodeChunk = copy.deepcopy(image_av_chunk)
    chunk.merge(chunk2)
    assert set(chunk.type_shape_table.keys()) == set([RGBImage, TerminateFlag])

    # NG
    depth_seq = ElementSequence([DepthImage(np.zeros((100, 100, 1))) for _ in range(10)])
    data = EpisodeData.from_seq_list([image_seq, depth_seq])
    chunk3 = MultiEpisodeChunk.from_data_list([data], with_intact_data=False)
    with pytest.raises(AssertionError):
        chunk.merge(chunk3)
