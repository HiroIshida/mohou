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


def get_subproject_dir(project_name: str, directory_name: str):
    dirname = osp.join(get_project_dir(project_name), directory_name)
    if not osp.exists(dirname):
        os.makedirs(dirname)
    return dirname


def resolve_file_name(obj_type: Type, project_name: str, postfix: Optional[str] = None) -> str:
    dir_path = get_project_dir(project_name)
    if postfix is None:
        file_name = osp.join(dir_path, obj_type.__name__ + '.pkl')
    else:
        file_name = osp.join(dir_path, obj_type.__name__ + '-' + postfix + '.pkl')
    return file_name


DataT = TypeVar('DataT')


def load_object(obj_type: Type[DataT], project_name: str, postfix: Optional[str] = None) -> DataT:
    file_name = resolve_file_name(obj_type, project_name, postfix)
    logger.info('load pickle from {}'.format(file_name))
    with open(file_name, 'rb') as f:
        obj = pickle.load(f)
    return obj


def dump_object(obj: Any, project_name: str, postfix: Optional[str] = None) -> None:
    file_name = resolve_file_name(obj.__class__, project_name, postfix)
    logger.info('dump pickle to {}'.format(file_name))
    with open(file_name, 'wb') as f:
        pickle.dump(obj, f)
