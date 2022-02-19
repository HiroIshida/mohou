import logging
import os
import os.path as osp
import pickle
from typing import Any, Optional, Type, TypeVar

logger = logging.getLogger(__name__)


def get_data_dir() -> str:
    dirname = osp.expanduser('~/.mohou')
    if not osp.exists(dirname):
        os.makedirs(dirname)
    return dirname


def get_project_dir(project_name: str) -> str:
    dirname = osp.join(get_data_dir(), project_name)
    if not osp.exists(dirname):
        os.makedirs(dirname)
    return dirname


def dump_object(obj: Any, project_name: str, postfix: Optional[str] = None) -> None:
    dir_path = get_project_dir(project_name)
    if postfix is None:
        file_name = osp.join(dir_path, obj.__class__.__name__ + '.pkl')
    else:
        file_name = osp.join(dir_path, obj.__class__.__name__ + '-' + postfix + '.pkl')
    logger.info('dump pickle to {}'.format(file_name))
    with open(file_name, 'wb') as f:
        pickle.dump(obj, f)


DataT = TypeVar('DataT')


def load_object(obj_type: Type[DataT], project_name: str) -> DataT:
    dir_path = get_project_dir(project_name)
    file_name = osp.join(dir_path, obj_type.__name__ + '.pkl')
    logger.info('load pickle from {}'.format(file_name))
    with open(file_name, 'rb') as f:
        obj = pickle.load(f)
    return obj
