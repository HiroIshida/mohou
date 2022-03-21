import pytest

import dataclasses
import uuid
import numpy as np
from mohou.file import remove_project, load_object, load_objects, dump_object


@dataclasses.dataclass
class SampleClass:
    data: np.ndarray


@pytest.fixture(scope='module')
def tmp_project_name():
    return 'pytest' + str(uuid.uuid4())


def test_dump_and_load_object(tmp_project_name):
    a = SampleClass(np.random.randn(10, 10))
    dump_object(a, tmp_project_name)
    b = load_object(SampleClass, tmp_project_name)
    np.testing.assert_almost_equal(a.data, b.data)
    remove_project(tmp_project_name)


def test_load_objects(tmp_project_name):
    a = SampleClass(np.random.randn(10, 10))
    b = SampleClass(np.random.randn(10, 10))
    dump_object(a, tmp_project_name, str(uuid.uuid4()))
    dump_object(b, tmp_project_name, str(uuid.uuid4()))

    objects = load_objects(SampleClass, tmp_project_name)
    np.testing.assert_almost_equal(a.data + b.data, objects[0].data + objects[1].data)
