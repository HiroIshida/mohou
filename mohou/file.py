import logging
import datetime
import shutil
import os
from pathlib import Path
import re
import pickle
from typing import Any, List, Optional, Type, TypeVar, Union

logger = logging.getLogger(__name__)


def get_data_path() -> Path:
    path = Path('~/.mohou').expanduser()
    path.mkdir(exist_ok=True)
    return path


def get_project_path(project_name: str) -> Path:
    path = get_data_path()
    project_dir_path = path / project_name
    project_dir_path.mkdir(exist_ok=True)
    return project_dir_path


def get_subproject_path(project_name: str, subpath: Union[str, Path]):
    if isinstance(subpath, str):
        subpath = Path(subpath)

    path = get_project_path(project_name)
    path = path / subpath
    path.mkdir(parents=True, exist_ok=True)
    return path


def remove_project(project_name: str) -> None:
    path = get_project_path(project_name)
    shutil.rmtree(str(path))


def resolve_file_path(
        obj_type: Type,
        project_name: str,
        postfix: Optional[str] = None,
        subpath: Optional[Path] = None) -> Path:

    dir_path = get_project_path(project_name)

    if subpath is not None:
        dir_path = dir_path / subpath

    dir_path.mkdir(parents=True, exist_ok=True)

    if postfix is None:
        file_path = dir_path / (obj_type.__name__ + '.pkl')
    else:
        file_path = dir_path / (obj_type.__name__ + '-' + postfix + '.pkl')
    return file_path


DataT = TypeVar('DataT')


def load_object(
        obj_type: Type[DataT],
        project_name: str,
        postfix: Optional[str] = None,
        subpath: Optional[Path] = None) -> DataT:

    file_path = resolve_file_path(
        obj_type, project_name, postfix=postfix, subpath=subpath)
    time_stamp = os.path.getmtime(str(file_path))
    modified_time = datetime.datetime.fromtimestamp(time_stamp)

    logger.info('load pickle from {} (modification time is {})'.format(
        str(file_path), modified_time))

    with file_path.open(mode='rb') as f:
        obj = pickle.load(f)

    return obj


def load_objects(
        obj_type: Type[DataT],
        project_name: str,
        postfix: Optional[str] = None,
        subpath: Optional[Path] = None) -> List[DataT]:

    file_path = resolve_file_path(
        obj_type, project_name, postfix=postfix, subpath=subpath)
    file_name_common = str(file_path)
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
    if len(obj_list) == 0:
        raise FileNotFoundError

    return obj_list


def dump_object(
        obj: Any,
        project_name: str,
        postfix: Optional[str] = None,
        subpath: Optional[Path] = None) -> None:

    file_path = resolve_file_path(
        obj.__class__, project_name, postfix=postfix, subpath=subpath)
    logger.info('dump pickle to {}'.format(file_path))

    # Not using with statement to use custom exception handling
    file_name = str(file_path)
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
