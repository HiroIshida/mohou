import argparse
from pathlib import Path
from typing import Optional

from mohou.encoding_rule import EncodingRule
from mohou.file import get_project_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-pp", type=str, help="project path. preferred over pn.")
    parser.add_argument("-pn", type=str, help="project name")
    args = parser.parse_args()
    project_name: Optional[str] = args.pn
    project_path_str: Optional[str] = args.pp

    if project_path_str is None:
        project_path = get_project_path(project_name)
    else:
        project_path = Path(project_path_str)

    rule = EncodingRule.create_default(project_path)
    rule.scale_balancer.dump(project_path)
