import argparse

from mohou.default import create_chimera_propagator, create_default_propagator
from mohou.script_utils import visualize_lstm_propagation
from mohou.setting import setting

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-pn", type=str, default=setting.primary_project_name, help="project name")
    parser.add_argument("-n", type=int, default=150, help="number of visualization")
    parser.add_argument("--chimera", action="store_true", help="use chimera")

    args = parser.parse_args()
    project_name = args.pn
    n_prop = args.n

    if args.chimera:
        propagator = create_chimera_propagator(project_name)
    else:
        propagator = create_default_propagator(project_name)

    visualize_lstm_propagation(project_name, propagator, n_prop)
