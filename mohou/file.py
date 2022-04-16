import logging
import os
import shutil
import os.path as osp
import re
import pickle
from typing import Any, List, Optional, Type, TypeVar

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


def remove_project(project_name: str) -> None:
    project_dir = get_project_dir(project_name)
    shutil.rmtree(project_dir)


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


def load_objects(obj_type: Type[DataT], project_name: str, postfix: Optional[str] = None) -> List[DataT]:
    file_name_common = resolve_file_name(obj_type, project_name, postfix)
    base, file_name_common_local = os.path.split(file_name_common)
    _, ext = os.path.splitext(file_name_common_local)

    obj_list: List[DataT] = []
    for file_name in os.listdir(base):
        file_name_common_local_noext, _ = os.path.splitext(file_name_common_local)
        result = re.match(r'{}*.'.format(file_name_common_local_noext), file_name)
        if result is not None:
            whole_name = os.path.join(base, file_name)
            with open(whole_name, 'rb') as f:
                obj = pickle.load(f)
            obj_list.append(obj)
    return obj_list


def dump_object(obj: Any, project_name: str, postfix: Optional[str] = None) -> None:
    file_name = resolve_file_name(obj.__class__, project_name, postfix)
    logger.info('dump pickle to {}'.format(file_name))

    # Not using with statement to use custom exception handling
    f = open(file_name, 'wb')
    try:
        pickle.dump(obj, f)
        f.close()
    except KeyboardInterrupt:
        f.close()
        logger.info('got keyboard interuppt. but let me dump the object...')
        with open(file_name, 'wb') as f:
            pickle.dump(obj, f)
    except Exception as e:
        f.close()
        raise e
