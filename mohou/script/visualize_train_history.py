import argparse
from mohou.script_utils import visualize_train_histories

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-pn', type=str, default='kuka_reaching', help='project name')
    args = parser.parse_args()
    project_name = args.pn
    visualize_train_histories(project_name)
