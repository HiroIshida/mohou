import argparse
from pathlib import Path

from mohou.default import create_chimera_propagator, create_default_propagator
from mohou.file import get_project_path
from mohou.propagator import PBLSTMPropagator, Propagator, PropagatorBase
from mohou.script_utils import visualize_lstm_propagation
from mohou.setting import setting


def get_propagator(use_chimera: bool, use_pb: bool) -> PropagatorBase:
    if use_chimera:
        assert not use_pb
        return create_chimera_propagator(project_path)
    elif args.use_pb:
        propagator = create_default_propagator(project_path, prop_type=PBLSTMPropagator)
        propagator.set_pb_to_zero()
        return propagator
    else:
        return create_default_propagator(project_path, prop_type=Propagator)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-pn", type=str, default=setting.primary_project_name, help="project name")
    parser.add_argument("-pp", type=str, help="project path. preferred over pn.")
    parser.add_argument("-n", type=int, default=150, help="number of visualization")
    parser.add_argument("--chimera", action="store_true", help="use chimera")
    parser.add_argument("--use_pb", action="store_true", help="use PBLSTMPropagator")

    args = parser.parse_args()
    project_name: str = args.pn
    project_path_str: str = args.pp
    n_prop: int = args.n

    if project_path_str is None:
        project_path = get_project_path(project_name)
    else:
        project_path = Path(project_path_str)

    propagator = get_propagator(args.chimera, args.use_pb)
    visualize_lstm_propagation(project_path, propagator, n_prop)
