import copy
import os
import pathlib
import pickle
import tempfile
from dataclasses import dataclass
from typing import Type

import numpy as np
import pytest
from test_file import tmp_project_name  # noqa

from mohou.file import remove_project
from mohou.types import (
    AngleVector,
    BundleSpec,
    DepthImage,
    ElementDict,
    ElementSequence,
    EpisodeBundle,
    EpisodeData,
    GrayImage,
    GripperState,
    HashableMixin,
    MetaData,
    PrimitiveImageBase,
    RGBDImage,
    RGBImage,
    TerminateFlag,
    TimeStampSequence,
    VectorBase,
    _bundle_cache,
    extract_contour_by_laplacian,
)


@pytest.fixture(scope="session")
def sample_image_path():
    return os.path.join(pathlib.Path(__file__).resolve().parent, "data", "sample.png")


@dataclass
class Hoge(HashableMixin):
    a: int


@dataclass
class Fuga(HashableMixin):
    b: int


def test_hashable_mixin():

    h = Hoge(1)
    f = Fuga(1)
    assert f.hash_value != h.hash_value

    h2 = Hoge(1)
    assert h.hash_value == h2.hash_value


def test_metadata():
    m1 = MetaData({"hoge": "hogehoge"})
    m2 = MetaData({"hoge": "hogehoge"})
    m3 = MetaData({"fuga": "fugafuaga"})
    assert m1.hash_value == m2.hash_value
    with pytest.raises(AssertionError):
        assert m1.hash_value == m3.hash_value


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
    a = RGBImage.dummy_from_shape((100, 100))
    assert a.shape == (100, 100, 3)
    elem_seq = ElementSequence[RGBImage]([a])
    assert elem_seq.elem_shape == (100, 100, 3)
    assert elem_seq.elem_type == RGBImage

    # check inconsistent shape
    with pytest.raises(AssertionError):
        a = RGBImage.dummy_from_shape((100, 100))
        b = RGBImage.dummy_from_shape((100, 101))
        ElementSequence[RGBImage]([a, b])

    # check inconsistent
    with pytest.raises(AssertionError):
        a = RGBImage.dummy_from_shape((100, 100))
        b = DepthImage.dummy_from_shape((100, 100))
        ElementSequence([a, b])

    # test dump and load
    with tempfile.TemporaryDirectory() as dname:
        dpath = pathlib.Path(dname)
        elem_seq.dump(dpath)
        elem_seq_again = ElementSequence.load(dpath, RGBImage)
        assert elem_seq == elem_seq_again


def test_episode_data():
    # creation
    image_seq = ElementSequence([RGBImage.dummy_from_shape((100, 100)) for _ in range(10)])
    av_seq = ElementSequence([AngleVector(np.random.randn(10)) for _ in range(10)])
    ts_seq = TimeStampSequence([i for i in range(10)])
    episode = EpisodeData.from_seq_list(
        [image_seq, av_seq], timestamp_seq=ts_seq, metadata=MetaData({"id": "hogehoge"})
    )

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
    assert episode_partial.metadata.hash_value == episode.metadata.hash_value

    episode_partial = episode[[2, 6]]
    assert isinstance(episode_partial, EpisodeData)  # access by indices
    assert episode_partial[0][RGBImage] == image_seq[2]
    assert episode_partial[-1][RGBImage] == image_seq[6]

    # test slice_by_time
    episode_partial = episode.slice_by_time(1.2, 9.0, 5.0)
    assert episode_partial.metadata.hash_value == episode.metadata.hash_value
    assert len(episode_partial) == 8
    flag_seq = episode_partial.get_sequence_by_type(TerminateFlag)

    for i in range(8):
        episode_partial[i] == episode[i + 2]
    for i in range(0, 3):
        assert flag_seq[i] == TerminateFlag.from_bool(False)
    for i in range(3, 8):
        assert flag_seq[i] == TerminateFlag.from_bool(True)

    # test dump and load
    with tempfile.TemporaryDirectory() as dname:
        dpath = pathlib.Path(dname)
        episode.dump(dpath)
        episode_again = EpisodeData.load(dpath)
        assert episode == episode_again


def test_episode_data_set_sequence():
    av_seq = ElementSequence([AngleVector(np.random.randn(10)) for _ in range(10)])
    episode = EpisodeData.from_seq_list([av_seq])

    im_seq = ElementSequence([RGBImage.dummy_from_shape((10, 10)) for _ in range(10)])
    episode.set_sequence(RGBImage, im_seq)  # ok

    # check inconsistent type
    with pytest.raises(AssertionError):
        episode.set_sequence(RGBImage, av_seq)  # NG

    # check inconsistent length
    with pytest.raises(AssertionError):
        av_seq = ElementSequence([AngleVector(np.random.randn(10)) for _ in range(11)])
        episode.set_sequence(AngleVector, av_seq)  # NG

    # check non-primitive elem
    with pytest.raises(AssertionError):
        rgbd_seq = ElementSequence([RGBDImage.dummy_from_shape((10, 10)) for _ in range(10)])
        episode.set_sequence(RGBDImage, rgbd_seq)  # NG


def test_episode_data_assertion_different_size():
    image_seq = ElementSequence([RGBImage.dummy_from_shape((100, 100)) for _ in range(3)])
    av_seq = ElementSequence([AngleVector(np.zeros(10)) for _ in range(10)])

    with pytest.raises(AssertionError):
        EpisodeData.from_seq_list([image_seq, av_seq])


def test_episode_data_assertion_type_inconsitency():
    image_seq = ElementSequence([RGBImage.dummy_from_shape((100, 100)) for _ in range(10)])

    with pytest.raises(AssertionError):
        EpisodeData.from_seq_list([image_seq, image_seq])


def test_two_bundle_consistency():
    def create_edata(n_length):
        av_seq = ElementSequence([AngleVector(np.random.randn(10)) for _ in range(n_length)])
        data = EpisodeData.from_seq_list([av_seq])
        return data

    lst = [create_edata(5) for _ in range(100)]
    bundle = EpisodeBundle.from_data_list(copy.deepcopy(lst), shuffle=True)
    bundle2 = EpisodeBundle.from_data_list(copy.deepcopy(lst), shuffle=True)
    assert pickle.dumps(bundle) == pickle.dumps(bundle2)


@pytest.fixture(scope="session")
def image_bundle():
    def create_edata(n_length):
        image_seq = ElementSequence(
            [RGBImage.dummy_from_shape((100, 100)) for _ in range(n_length)]
        )
        data = EpisodeData.from_seq_list([image_seq])
        return data

    lst = [create_edata(10) for _ in range(20)]
    bundle = EpisodeBundle.from_data_list(lst, meta_data=MetaData({"hoge": 1.0}))
    return bundle


@pytest.fixture(scope="session")
def image_av_bundle():
    def create_edata(n_length):
        image_seq = ElementSequence([RGBImage.dummy_from_shape((28, 28)) for _ in range(n_length)])
        av_seq = ElementSequence([AngleVector(np.random.randn(10)) for _ in range(n_length)])
        data = EpisodeData.from_seq_list([image_seq, av_seq])
        return data

    lst = [create_edata(10) for _ in range(20)]
    bundle = EpisodeBundle.from_data_list(lst, meta_data=MetaData({"hoge": 1.0}))
    return bundle


@pytest.fixture(scope="session")
def image_av_bundle_uneven():
    def create_edata(n_length):
        image_seq = ElementSequence([RGBImage.dummy_from_shape((28, 28)) for _ in range(n_length)])
        av_seq = ElementSequence([AngleVector(np.zeros(10)) for _ in range(n_length)])
        data = EpisodeData.from_seq_list([image_seq, av_seq])
        return data

    lst = [create_edata(10) for _ in range(20)]
    lst.append(create_edata(13))
    bundle = EpisodeBundle.from_data_list(lst, shuffle=False, leave_untouch_episode=False)
    return bundle


def test_bundle_spec():
    types = {RGBImage: (100, 100, 3), AngleVector: (7,)}
    extra_info = {"hz": 20, "author": "HiroIshida"}
    spec = BundleSpec(10, 5, 10, types, meta_data=extra_info)
    spec_reconstructed = BundleSpec.from_dict(spec.to_dict())
    assert pickle.dumps(spec) == pickle.dumps(spec_reconstructed)


def test_multi_episode_bundle(image_av_bundle, image_bundle, tmp_project_name):  # noqa
    bundle: EpisodeBundle = image_av_bundle
    assert set(bundle.types()) == set([AngleVector, RGBImage, TerminateFlag])

    bundle.dump(tmp_project_name)
    assert tmp_project_name not in _bundle_cache
    loaded = bundle.load(tmp_project_name)
    assert bundle == loaded
    assert (tmp_project_name, None) in _bundle_cache

    bundle.dump(tmp_project_name, postfix="without_tar", use_tar=False)
    loaded2 = EpisodeBundle.load(tmp_project_name, postfix="without_tar", use_tar=False)
    assert bundle == loaded2

    # test having multiple bundle in one project
    postfix = "extra"
    extra_bundle: EpisodeBundle = image_bundle
    extra_bundle.dump(tmp_project_name, postfix)
    extra_bundle_loaded = EpisodeBundle.load(tmp_project_name, postfix)
    assert extra_bundle == extra_bundle_loaded

    remove_project(tmp_project_name)


def test_multi_episode_bundle_assertion_type_inconsitency():
    image_seq = ElementSequence([RGBImage.dummy_from_shape((100, 100)) for _ in range(10)])
    av_seq = ElementSequence([AngleVector(np.zeros(10)) for _ in range(10)])
    depth_seq = ElementSequence([DepthImage(np.zeros((100, 100, 1))) for _ in range(10)])

    data1 = EpisodeData.from_seq_list([image_seq, av_seq])
    data2 = EpisodeData.from_seq_list([depth_seq, av_seq])

    with pytest.raises(AssertionError):
        EpisodeBundle.from_data_list([data1, data2])
