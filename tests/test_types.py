import os
import pathlib
import pickle
from typing import Type

import numpy as np
import pytest
from test_file import tmp_project_name  # noqa

from mohou.types import (
    AngleVector,
    ChunkSpec,
    DepthImage,
    ElementDict,
    ElementSequence,
    EpisodeData,
    GrayImage,
    GripperState,
    MultiEpisodeChunk,
    PrimitiveImageBase,
    RGBDImage,
    RGBImage,
    TerminateFlag,
    TimeStampSequence,
    VectorBase,
    _chunk_cache,
    extract_contour_by_laplacian,
)


@pytest.fixture(scope="session")
def sample_image_path():
    return os.path.join(pathlib.Path(__file__).resolve().parent, "data", "sample.png")


def test_elements():
    with pytest.raises(Exception):
        VectorBase(np.zeros(3))
    with pytest.raises(Exception):
        PrimitiveImageBase(np.zeros((3, 3)))


def test_gripper_state():
    for arr in [np.zeros(2), np.ones(2)]:
        gs = GripperState(arr.astype(bool))
        gs_reconstructed = GripperState.from_tensor(gs.to_tensor())
        assert gs == gs_reconstructed


def test_rdb_image_creation(sample_image_path):

    with pytest.raises(AssertionError):
        RGBImage(np.random.randint(0, 255, (100, 100), dtype=np.uint8))
    with pytest.raises(AssertionError):
        RGBImage(np.random.randn(100, 100, 3))

    RGBImage(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))

    RGBImage.from_file(sample_image_path)


def test_gray_image_creation():

    with pytest.raises(AssertionError):
        GrayImage(np.random.randint(0, 255, (100, 100), dtype=np.uint8))
    with pytest.raises(AssertionError):
        GrayImage(np.random.randint(0, 255, (100, 100, 2), dtype=np.uint8))
    with pytest.raises(AssertionError):
        GrayImage(np.random.randn(100, 100, 1))

    GrayImage(np.random.randint(0, 255, (100, 100, 1), dtype=np.uint8))


def test_depth_image_creation():

    with pytest.raises(AssertionError):
        DepthImage(np.random.randn(100, 100))
    with pytest.raises(AssertionError):
        DepthImage(np.random.randn(100, 100, 2))
    with pytest.raises(AssertionError):
        DepthImage(np.random.randint(0, 255, (100, 100, 1)))

    DepthImage(np.random.randn(100, 100, 1))


@pytest.mark.parametrize("T", [RGBImage, GrayImage, DepthImage])
def test_primitive_images(T: Type[PrimitiveImageBase]):
    img = T.dummy_from_shape((100, 100))
    img2 = T.from_tensor(img.to_tensor())
    assert img == img2

    with pytest.raises(AssertionError):
        assert img.randomize() == img2.randomize()

    img.to_rgb()
    assert img.shape == (100, 100, img.channel())
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
        assert im1 == im2

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


def test_extract_contour_by_laplacian(sample_image_path):
    # just a pipeline test
    rgb = RGBImage.from_file(sample_image_path)
    extract_contour_by_laplacian(rgb)


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
    ts_seq = TimeStampSequence([i for i in range(10)])
    episode = EpisodeData.from_seq_list([image_seq, av_seq], timestamp_seq=ts_seq)

    assert set(episode.types()) == set([AngleVector, RGBImage, TerminateFlag])

    # test __getitem__
    edict = episode[0]
    assert isinstance(edict, ElementDict)  # access by index
    assert edict[RGBImage] == image_seq[0]

    i_start = 2
    i_end = 5
    episode_partial = episode[i_start:i_end]
    assert isinstance(episode_partial, EpisodeData)  # access by slice
    assert episode_partial[0][RGBImage] == image_seq[i_start]
    assert episode_partial[-1][RGBImage] == image_seq[i_end - 1]

    episode_partial = episode[[2, 6]]
    assert isinstance(episode_partial, EpisodeData)  # access by indices
    assert episode_partial[0][RGBImage] == image_seq[2]
    assert episode_partial[-1][RGBImage] == image_seq[6]

    # test slice_by_time
    episode_partial = episode.slice_by_time(0, 7.0, 4.0)
    assert len(episode_partial) == 8
    flag_seq = episode_partial.get_sequence_by_type(TerminateFlag)
    for i in range(0, 4):
        assert flag_seq[i] == TerminateFlag.from_bool(False)
    for i in range(5, 8):
        assert flag_seq[i] == TerminateFlag.from_bool(True)


def test_episode_data_assertion_different_size():
    image_seq = ElementSequence([RGBImage.dummy_from_shape((100, 100)) for _ in range(3)])
    av_seq = ElementSequence([AngleVector(np.zeros(10)) for _ in range(10)])

    with pytest.raises(AssertionError):
        EpisodeData.from_seq_list([image_seq, av_seq])


def test_episode_data_assertion_type_inconsitency():
    image_seq = ElementSequence([RGBImage.dummy_from_shape((100, 100)) for _ in range(10)])

    with pytest.raises(AssertionError):
        EpisodeData.from_seq_list([image_seq, image_seq])


@pytest.fixture(scope="session")
def image_chunk():
    def create_edata(n_length):
        image_seq = ElementSequence(
            [RGBImage.dummy_from_shape((100, 100)) for _ in range(n_length)]
        )
        data = EpisodeData.from_seq_list([image_seq])
        return data

    lst = [create_edata(10) for _ in range(20)]
    chunk = MultiEpisodeChunk.from_data_list(lst)
    return chunk


@pytest.fixture(scope="session")
def image_av_chunk():
    def create_edata(n_length):
        image_seq = ElementSequence(
            [RGBImage.dummy_from_shape((100, 100)) for _ in range(n_length)]
        )
        av_seq = ElementSequence([AngleVector(np.zeros(10)) for _ in range(n_length)])
        data = EpisodeData.from_seq_list([image_seq, av_seq])
        return data

    lst = [create_edata(10) for _ in range(20)]
    chunk = MultiEpisodeChunk.from_data_list(lst)
    return chunk


@pytest.fixture(scope="session")
def image_av_chunk_uneven():
    def create_edata(n_length):
        image_seq = ElementSequence(
            [RGBImage.dummy_from_shape((100, 100)) for _ in range(n_length)]
        )
        av_seq = ElementSequence([AngleVector(np.zeros(10)) for _ in range(n_length)])
        data = EpisodeData.from_seq_list([image_seq, av_seq])
        return data

    lst = [create_edata(10) for _ in range(20)]
    lst.append(create_edata(13))
    chunk = MultiEpisodeChunk.from_data_list(lst, shuffle=False, with_intact_data=False)
    return chunk


def test_chunk_spec():
    types = {RGBImage: (100, 100, 3), AngleVector: (7,)}
    extra_info = {"hz": 20, "author": "HiroIshida"}
    spec = ChunkSpec(10, 5, 10, types, extra_info=extra_info)
    spec_reconstructed = ChunkSpec.from_dict(spec.to_dict())
    assert pickle.dumps(spec) == pickle.dumps(spec_reconstructed)


def test_multi_episode_chunk(image_av_chunk, image_chunk, tmp_project_name):  # noqa
    chunk: MultiEpisodeChunk = image_av_chunk
    assert set(chunk.types()) == set([AngleVector, RGBImage, TerminateFlag])

    chunk.dump(tmp_project_name)
    assert tmp_project_name not in _chunk_cache
    loaded = chunk.load(tmp_project_name)
    assert pickle.dumps(chunk) == pickle.dumps(loaded)
    assert (tmp_project_name, None) in _chunk_cache

    chunk_spec_loaded = MultiEpisodeChunk.load_spec(tmp_project_name)
    assert pickle.dumps(chunk.spec) == pickle.dumps(chunk_spec_loaded)

    # test having multiple chunk in one project
    postfix = "extra"
    extra_chunk: MultiEpisodeChunk = image_chunk
    extra_chunk.dump(tmp_project_name, postfix)
    extra_chunk_loaded = MultiEpisodeChunk.load(tmp_project_name, postfix)
    assert pickle.dumps(extra_chunk) == pickle.dumps(extra_chunk_loaded)

    extra_chunk_spec_loaded = MultiEpisodeChunk.load_spec(tmp_project_name, postfix)
    assert pickle.dumps(extra_chunk.spec) == pickle.dumps(extra_chunk_spec_loaded)


def test_multi_episode_chunk_assertion_type_inconsitency():
    image_seq = ElementSequence([RGBImage.dummy_from_shape((100, 100)) for _ in range(10)])
    av_seq = ElementSequence([AngleVector(np.zeros(10)) for _ in range(10)])
    depth_seq = ElementSequence([DepthImage(np.zeros((100, 100, 1))) for _ in range(10)])

    data1 = EpisodeData.from_seq_list([image_seq, av_seq])
    data2 = EpisodeData.from_seq_list([depth_seq, av_seq])

    with pytest.raises(AssertionError):
        MultiEpisodeChunk.from_data_list([data1, data2])
