import argparse

from mohou.default import create_chimera_propagator, create_default_propagator
from mohou.file import get_project_path
from mohou.script_utils import visualize_lstm_propagation
from mohou.setting import setting

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-pn", type=str, default=setting.primary_project_name, help="project name")
    parser.add_argument("-n", type=int, default=150, help="number of visualization")
    parser.add_argument("--chimera", action="store_true", help="use chimera")

    args = parser.parse_args()
    project_name: str = args.pn
    n_prop: int = args.n

    project_path = get_project_path(project_name)

    if args.chimera:
        propagator = create_chimera_propagator(project_path)
    else:
        propagator = create_default_propagator(project_path)

    visualize_lstm_propagation(project_name, propagator, n_prop)
