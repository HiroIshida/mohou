import argparse
from pathlib import Path

from mohou.default import create_default_chimera_propagator, create_default_propagator
from mohou.file import get_project_path
from mohou.propagator import PBLSTMPropagator, Propagator, PropagatorBase
from mohou.script_utils import visualize_lstm_propagation
from mohou.setting import setting


def get_propagator(project_path: Path, use_pb: bool, use_chimera: bool) -> PropagatorBase:
    assert not (use_pb and use_chimera), "currently chimera does not support pb"

    if use_pb:
        propagator = create_default_propagator(project_path, prop_type=PBLSTMPropagator)
        propagator.set_pb_to_zero()
        return propagator
    elif use_chimera:
        return create_default_chimera_propagator(project_path)
    else:
        return create_default_propagator(project_path, prop_type=Propagator)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-pn", type=str, default=setting.primary_project_name, help="project name")
    parser.add_argument("-pp", type=str, help="project path. preferred over pn.")
    parser.add_argument("-n", type=int, default=150, help="number of visualization")
    parser.add_argument("--use_pb", action="store_true", help="use PBLSTMPropagator")
    parser.add_argument("--chimera", action="store_true", help="use Chimera model")

    args = parser.parse_args()
    project_name: str = args.pn
    project_path_str: str = args.pp
    n_prop: int = args.n

    if project_path_str is None:
        project_path = get_project_path(project_name)
    else:
        project_path = Path(project_path_str)

    propagator = get_propagator(project_path, args.use_pb, args.chimera)
    visualize_lstm_propagation(project_path, propagator, n_prop)
