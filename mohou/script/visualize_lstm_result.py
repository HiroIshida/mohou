import argparse
from pathlib import Path

from mohou.file import get_project_path
from mohou.propagator import (
    ChimeraPropagator,
    LSTMPropagator,
    LSTMPropagatorBase,
    PBLSTMPropagator,
)
from mohou.script_utils import visualize_lstm_propagation
from mohou.setting import setting


def get_propagator(project_path: Path, use_pb: bool, use_chimera: bool) -> LSTMPropagatorBase:
    assert not (use_pb and use_chimera), "currently chimera does not support pb"

    if use_pb:
        return PBLSTMPropagator.create_default(project_path)
    elif use_chimera:
        return ChimeraPropagator.create_default(project_path)
    else:
        return LSTMPropagator.create_default(project_path)


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
