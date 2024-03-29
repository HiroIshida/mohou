import dataclasses
import pathlib
import uuid

import numpy as np
import pytest

from mohou.file import (
    create_project_dir,
    dump_object,
    load_object,
    load_objects,
    remove_project,
)


@dataclasses.dataclass
class SampleClass:
    data: np.ndarray


@pytest.fixture(scope="module")
def tmp_project_name():
    name = "pytest-" + str(uuid.uuid4())
    return name


def test_dump_and_load_object(tmp_project_name):
    create_project_dir(tmp_project_name)
    a = SampleClass(np.random.randn(10, 10))
    dump_object(a, tmp_project_name)
    b = load_object(SampleClass, tmp_project_name)
    np.testing.assert_almost_equal(a.data, b.data)

    remove_project(tmp_project_name)


def test_dump_and_load_object_subdir(tmp_project_name):
    create_project_dir(tmp_project_name)
    a = SampleClass(np.random.randn(10, 10))

    subpath = pathlib.Path("hoge/hoge")
    dump_object(a, tmp_project_name, subpath=subpath)
    b = load_object(SampleClass, tmp_project_name, subpath=subpath)
    np.testing.assert_almost_equal(a.data, b.data)

    remove_project(tmp_project_name)


def test_load_objects(tmp_project_name):
    create_project_dir(tmp_project_name)
    a = SampleClass(np.random.randn(10, 10))
    b = SampleClass(np.random.randn(10, 10))
    dump_object(a, tmp_project_name, str(uuid.uuid4()))
    dump_object(b, tmp_project_name, str(uuid.uuid4()))

    objects = load_objects(SampleClass, tmp_project_name)
    np.testing.assert_almost_equal(a.data + b.data, objects[0].data + objects[1].data)

    remove_project(tmp_project_name)


def test_load_objects_partial_match(tmp_project_name):
    create_project_dir(tmp_project_name)
    a = SampleClass(np.random.randn(10, 10))
    b = SampleClass(np.random.randn(10, 10))
    common_uuid = str(uuid.uuid4())
    dump_object(a, tmp_project_name, common_uuid + str(uuid.uuid4()))
    dump_object(b, tmp_project_name, common_uuid + str(uuid.uuid4()))

    objects = load_objects(SampleClass, tmp_project_name, postfix=common_uuid)
    assert len(objects) == 2

    with pytest.raises(FileNotFoundError):
        wrong_uuid = str(uuid.uuid4())
        objects = load_objects(SampleClass, tmp_project_name, postfix=wrong_uuid)

    remove_project(tmp_project_name)


def test_load_objects_subdir(tmp_project_name):
    create_project_dir(tmp_project_name)
    subpath = pathlib.Path("hoge/hoge")
    a = SampleClass(np.random.randn(10, 10))
    b = SampleClass(np.random.randn(10, 10))
    dump_object(a, tmp_project_name, str(uuid.uuid4()), subpath=subpath)
    dump_object(b, tmp_project_name, str(uuid.uuid4()), subpath=subpath)

    objects = load_objects(SampleClass, tmp_project_name, subpath=subpath)
    np.testing.assert_almost_equal(a.data + b.data, objects[0].data + objects[1].data)

    remove_project(tmp_project_name)


def test_load_objects_with_postfix(tmp_project_name):
    create_project_dir(tmp_project_name)
    postfix = "hogehoge"
    a = SampleClass(np.random.randn(10, 10))
    b = SampleClass(np.random.randn(10, 10))
    dump_object(a, tmp_project_name, postfix + str(uuid.uuid4()))
    dump_object(b, tmp_project_name, postfix + str(uuid.uuid4()))

    objects = load_objects(SampleClass, tmp_project_name, postfix)
    np.testing.assert_almost_equal(a.data + b.data, objects[0].data + objects[1].data)

    remove_project(tmp_project_name)


def test_filenotfounderrro_in_loading(tmp_project_name):
    create_project_dir(tmp_project_name)

    class HogeHoge:
        pass

    with pytest.raises(FileNotFoundError):
        load_object(HogeHoge, tmp_project_name)

    with pytest.raises(FileNotFoundError):
        load_objects(HogeHoge, tmp_project_name)

    remove_project(tmp_project_name)
