import functools
import pathlib
import queue
import subprocess
import types
from typing import Any, Callable, Iterator, List, Type, TypeVar, Union, cast

import numpy as np
import PIL
import torch
from torch.utils.data import Dataset, random_split

AnyT = TypeVar("AnyT", bound=Any)


class DataclassLightMixin:
    # sometimes, dataclass is too feature-rich and because of that
    # cause troubles. For example, dataclass is not straightly
    # incorpolated with abstract property.
    # To avoid this, this class provides minimalistic dataclass
    # implementation.
    def __init__(self, *args):
        assert len(args) == len(self.__annotations__)
        for key, arg in zip(self.__annotations__.keys(), args):
            setattr(self, key, arg)

    def __repr__(self) -> str:
        out = ""
        for key in self.__annotations__.keys():
            val = getattr(self, key)
            out += "{}: {} \n".format(key, repr(val))
        return out

    def __str__(self) -> str:
        return self.__repr__()


def abstract_attribute(obj: Callable[[Any], AnyT] = None) -> AnyT:
    class DummyAttribute:
        pass

    _obj = cast(Any, obj)
    if obj is None:
        _obj = DummyAttribute()
    _obj.__is_abstract_attribute__ = True
    return cast(AnyT, _obj)


def get_all_concrete_leaftypes(root: Type) -> List[Type]:
    concrete_types: List[Type] = []
    q = queue.Queue()  # type: ignore
    q.put(root)
    while not q.empty():
        t: Type = q.get()
        if len(t.__subclasses__()) == 0:
            concrete_types.append(t)

        for st in t.__subclasses__():
            q.put(st)
    return list(set(concrete_types))


def flatten_lists(list_list: List[List[AnyT]]) -> List[AnyT]:
    return functools.reduce(lambda xs, ys: xs + ys, list_list)


def get_bound_list(dims: List[int]) -> List[slice]:
    bound_list = []
    head = 0
    for dim in dims:
        bound_list.append(slice(head, head + dim))
        head += dim
    return bound_list


def splitting_slices(n_elem_list: List[int]) -> Iterator[slice]:
    head = 0
    for n_elem in n_elem_list:
        tail = head + n_elem
        yield slice(head, tail)
        head = tail


SequenceT = TypeVar("SequenceT")  # TODO(HiroIshida) bound?


def split_sequence(seq: SequenceT, n_elem_list: List[int]) -> Iterator[SequenceT]:
    for sl in splitting_slices(n_elem_list):
        yield seq[sl]  # type: ignore


def detect_device() -> torch.device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device


def split_with_ratio(dataset: Dataset, valid_ratio: float = 0.1):
    """split dataset into train and validation dataset with specified ratio"""

    n_total = len(dataset)  # type: ignore
    n_validate = int(valid_ratio * n_total)
    ds_train, ds_validate = random_split(dataset, [n_total - n_validate, n_validate])
    return ds_train, ds_validate


def assert_equal_with_message(given: AnyT, expected: Union[AnyT, List[Any]], elem_name: str):
    message = "{0}: given {1}, exepcted {2}".format(elem_name, given, expected)
    if isinstance(expected, list):
        assert given in expected, message
    else:
        assert given == expected, message


def assert_isinstance_with_message(given: Any, expected: Type):
    assert isinstance(given, expected), "{0}: given {1}, exepcted {2}".format(
        "not isinstance", given, expected
    )


def assert_seq_list_list_compatible(seq_llist: List[List[Any]]):
    if not __debug__:
        return
    seqlen_list_reference = [len(seq) for seq in seq_llist[0]]  # first seq_list as ref
    for seq_list in seq_llist:
        seqlen_list = [len(seq) for seq in seq_list]
        assert seqlen_list == seqlen_list_reference


def canvas_to_ndarray(fig, resize_pixel=None) -> np.ndarray:
    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep="")
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    if resize_pixel is None:
        return data
    img = PIL.Image.fromarray(data)
    img_resized = img.resize(resize_pixel)
    data_resized = np.asarray(img_resized, dtype=np.uint8)
    return data_resized


def log_text_with_box(logger, text: str) -> None:
    box_width = max(40, len(text) * 3)
    inner_text = "=" * 10 + " " + text.capitalize() + " "
    inner_text += "=" * (box_width - len(inner_text))

    logger.info("=" * box_width)
    logger.info(inner_text)
    logger.info("=" * box_width)


def log_package_version_info(logger, module: types.ModuleType) -> None:
    def log_line_by_line(multiline_text: str):
        for line in multiline_text.splitlines():
            logger.info(line)

    init_file_name: str = module.__file__  # type: ignore
    is_site_package = "site-packages" in init_file_name
    log_text_with_box(logger, "version check")
    if is_site_package:
        logger.info("version: {}".format(module.__version__))  # type: ignore
    else:
        repo_path = pathlib.Path(init_file_name).parent.parent

        # git log
        command_git_log = "cd {}; git --no-pager log --oneline | head -20".format(str(repo_path))
        proc = subprocess.run(command_git_log, stdout=subprocess.PIPE, shell=True)
        git_log_stdout = proc.stdout.decode("utf8")
        logger.info(command_git_log)
        log_line_by_line(git_log_stdout)

        # git diff
        log_text_with_box(logger, "git diff check")
        command_git_diff = "cd {}; git --no-pager diff".format(str(repo_path))
        proc = subprocess.run(command_git_diff, stdout=subprocess.PIPE, shell=True)
        git_diff_stdout = proc.stdout.decode("utf8")
        logger.info(command_git_diff)
        log_line_by_line(git_diff_stdout)
