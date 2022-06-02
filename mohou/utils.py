import queue
from typing import Any, List, Iterator, TypeVar, Union, Type, Sequence

import numpy as np
import PIL
import torch
from torch.utils.data import Dataset, random_split


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


def splitting_slices(n_elem_list: List[int]) -> Iterator[slice]:
    head = 0
    for n_elem in n_elem_list:
        tail = head + n_elem
        yield slice(head, tail)
        head = tail


SequenceT = TypeVar('SequenceT')  # TODO(HiroIshida) bound?


def split_sequence(seq: SequenceT, n_elem_list: List[int]) -> Iterator[SequenceT]:
    for sl in splitting_slices(n_elem_list):
        yield seq[sl]  # type: ignore


def detect_device() -> torch.device:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return device


def split_with_ratio(dataset: Dataset, valid_ratio: float = 0.1):
    """split dataset into train and validation dataset with specified ratio"""

    n_total = len(dataset)  # type: ignore
    n_validate = int(valid_ratio * n_total)
    ds_train, ds_validate = random_split(dataset, [n_total - n_validate, n_validate])
    return ds_train, ds_validate


AnyT = TypeVar('AnyT', bound=Any)


def assert_with_message(given: AnyT, expected: Union[AnyT, List[Any]], elem_name: str):
    message = '{0}: given {1}, exepcted {2}'.format(elem_name, given, expected)
    if isinstance(expected, list):
        assert given in expected, message
    else:
        assert given == expected, message


def assert_isinstance_with_message(given: Any, expected: Type):
    message = '{0}: given {1}, exepcted {2}'.format('not isinstance', given, expected)
    assert isinstance(given, expected), message


def assert_two_sequences_same_length(seq_list1: Sequence[np.ndarray], seq_list2: Sequence[np.ndarray]):
    if __debug__:
        assert_with_message(len(seq_list1), len(seq_list2), 'seq_list length of lhs')
        for seq1, seq2 in zip(seq_list1, seq_list2):
            assert_with_message(len(seq1), len(seq2), 'seq length of lhs')


def canvas_to_ndarray(fig, resize_pixel=None) -> np.ndarray:
    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    if resize_pixel is None:
        return data
    img = PIL.Image.fromarray(data)
    img_resized = img.resize(resize_pixel)
    data_resized = np.asarray(img_resized, dtype=np.uint8)
    return data_resized
