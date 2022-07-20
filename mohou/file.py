import datetime
import logging
import os
import pickle
import re
import shutil
from pathlib import Path
from typing import Any, List, Optional, Type, TypeVar, Union

from mohou.setting import setting

logger = logging.getLogger(__name__)


def get_root_path() -> Path:
    path = setting.root_path
    path.mkdir(exist_ok=True)
    return path


def create_project_dir(project_name: str) -> None:
    root_path = get_root_path()
    project_path = root_path / project_name
    project_path.mkdir(exist_ok=True)


def get_project_path(project_name: Optional[str] = None) -> Path:
    root_path = get_root_path()
    if project_name is None:
        assert setting.primary_project_name is not None
        project_name = setting.primary_project_name
    project_dir_path = root_path / project_name
    if not project_dir_path.exists():
        raise FileNotFoundError
    return project_dir_path


def get_subproject_path(project_name: Optional[str], subpath: Union[str, Path]) -> Path:

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
    project_name: Optional[str] = None,
    postfix: Optional[str] = None,
    subpath: Optional[Path] = None,
) -> Path:

    dir_path = get_project_path(project_name)

    if subpath is not None:
        dir_path = dir_path / subpath

    dir_path.mkdir(parents=True, exist_ok=True)

    if postfix is None:
        file_path = dir_path / (obj_type.__name__ + ".pkl")
    else:
        file_path = dir_path / (obj_type.__name__ + "-" + postfix + ".pkl")
    return file_path


DataT = TypeVar("DataT")


def load_object(
    obj_type: Type[DataT],
    project_name: Optional[str] = None,
    postfix: Optional[str] = None,
    subpath: Optional[Path] = None,
) -> DataT:
    """load single pickle object"""

    file_path = resolve_file_path(obj_type, project_name, postfix=postfix, subpath=subpath)
    time_stamp = os.path.getmtime(str(file_path))
    modified_time = datetime.datetime.fromtimestamp(time_stamp)

    logger.info(
        "load pickle from {} (modification time is {})".format(str(file_path), modified_time)
    )

    with file_path.open(mode="rb") as f:
        obj = pickle.load(f)

    return obj


def load_objects(
    obj_type: Type[DataT],
    project_name: Optional[str] = None,
    postfix: Optional[str] = None,
    subpath: Optional[Path] = None,
) -> List[DataT]:
    """load multiple pickle objects (or could be single object)
    If postfix is specified, all the objects filename of which starts with the postfix will be loaded.

    For example, if obj_type = SomeClass and postfix is "12345", file names
    path_to_project/Someclass-12345.pkl
    path_to_project/Someclass-12345-jfoiwfjoi.pkl
    will be loaded but
    path_to_project/Someclass-54321.pkl
    will not be loaded.
    """

    file_path = resolve_file_path(obj_type, project_name, postfix=postfix, subpath=subpath)
    file_name_common = str(file_path)
    base, file_name_common_local = os.path.split(file_name_common)
    _, ext = os.path.splitext(file_name_common_local)

    obj_list: List[DataT] = []
    for file_name in os.listdir(base):
        file_name_common_local_noext, _ = os.path.splitext(file_name_common_local)
        result = re.match(r"{}*.".format(file_name_common_local_noext), file_name)
        if result is not None:
            whole_name = os.path.join(base, file_name)
            with open(whole_name, "rb") as f:
                obj = pickle.load(f)
            obj_list.append(obj)
    if len(obj_list) == 0:
        raise FileNotFoundError

    return obj_list


def dump_object(
    obj: Any,
    project_name: Optional[str] = None,
    postfix: Optional[str] = None,
    subpath: Optional[Path] = None,
) -> None:

    file_path = resolve_file_path(obj.__class__, project_name, postfix=postfix, subpath=subpath)
    logger.info("dump pickle to {}".format(file_path))

    # Not using with statement to use custom exception handling
    file_name = str(file_path)
    f = open(file_name, "wb")
    try:
        pickle.dump(obj, f)
        f.close()
    except KeyboardInterrupt:
        f.close()
        logger.info("got keyboard interuppt. but let me dump the object...")
        with open(file_name, "wb") as f:
            pickle.dump(obj, f)
    except Exception as e:
        f.close()
        raise e
