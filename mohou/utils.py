import logging
from logging import Logger
import os
import time
from typing import List, Iterator, TypeVar

import torch
from torch.utils.data import Dataset, random_split

from mohou.file import get_project_dir


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


def create_default_logger(project_name: str, prefix: str) -> Logger:
    timestr = "_" + time.strftime("%Y%m%d%H%M%S")
    log_dir = os.path.join(get_project_dir(project_name), 'log')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_file_name = os.path.join(log_dir, (prefix + timestr + '.log'))
    FORMAT = '[%(levelname)s] %(asctime)s %(name)s: %(message)s'
    logging.basicConfig(filename=log_file_name, format=FORMAT)
    logger = logging.getLogger('mohou')
    logger.setLevel(level=logging.INFO)

    log_sym_name = os.path.join(log_dir, ('latest_' + prefix + '.log'))
    logger.info('create log symlink :{0} => {1}'.format(log_file_name, log_sym_name))
    if os.path.islink(log_sym_name):
        os.unlink(log_sym_name)
    os.symlink(log_file_name, log_sym_name)
    return logger
