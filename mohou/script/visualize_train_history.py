import argparse
from pathlib import Path
from typing import Optional

from mohou.file import get_project_path
from mohou.script_utils import visualize_train_histories
from mohou.setting import setting

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-pn", type=str, default=setting.primary_project_name, help="project name")
    parser.add_argument("-pp", type=str, help="project path. preferred over pn")
    args = parser.parse_args()
    project_name: str = args.pn
    project_path_str: Optional[str] = args.pp

    if project_path_str is None:
        project_path = get_project_path(project_name)
    else:
        project_path = Path(project_path_str)

    visualize_train_histories(project_path)
