import argparse
from enum import Enum
from pathlib import Path

from mohou.file import get_project_path
from mohou.propagator import (
    ChimeraPropagator,
    DisentangleLSTMPropagator,
    LSTMPropagator,
    PBLSTMPropagator,
    PropagatorBase,
    ProportionalModelPropagator,
)
from mohou.script_utils import visualize_lstm_propagation
from mohou.setting import setting


class Propagatorselection(Enum):
    lstm = LSTMPropagator
    pblstm = PBLSTMPropagator
    chimera = ChimeraPropagator
    proportional = ProportionalModelPropagator
    disentangle = DisentangleLSTMPropagator


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-pn", type=str, default=setting.primary_project_name, help="project name")
    parser.add_argument("-pp", type=str, help="project path. preferred over pn.")
    parser.add_argument("-n", type=int, default=150, help="number of visualization")
    parser.add_argument("-model", type=str, default="lstm", help="select model")

    args = parser.parse_args()
    project_name: str = args.pn
    project_path_str: str = args.pp
    n_prop: int = args.n
    model_str: str = args.model

    if project_path_str is None:
        project_path = get_project_path(project_name)
    else:
        project_path = Path(project_path_str)

    prop_type: PropagatorBase = Propagatorselection[model_str].value
    propagator = prop_type.create_default(project_path)
    visualize_lstm_propagation(project_path, propagator, n_prop)
