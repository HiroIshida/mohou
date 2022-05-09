import argparse

from mohou.default import create_default_propagator
from mohou.script_utils import visualize_lstm_propagation

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-pn', type=str, default='kuka_reaching', help='project name')
    parser.add_argument('-n', type=int, default=150, help='number of visualization')

    args = parser.parse_args()
    project_name = args.pn
    n_prop = args.n

    propagator = create_default_propagator(project_name)
    visualize_lstm_propagation(project_name, propagator, n_prop)
