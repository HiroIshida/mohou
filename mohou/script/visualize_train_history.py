import argparse
import os
import re
import pickle
import matplotlib.pyplot as plt
from mohou.file import get_project_dir, get_subproject_dir
from mohou.trainer import TrainCache

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-pn', type=str, default='kuka_reaching', help='project name')
    args = parser.parse_args()
    project_name = args.pn

    project_dir = get_project_dir(project_name)
    fnames = os.listdir(project_dir)

    plot_dir = get_subproject_dir(project_name, 'train_history')
    for fname in fnames:
        m = re.match(r'.*TrainCache.*', fname)
        if m is not None:
            pickle_file = os.path.join(project_dir, fname)

            with open(pickle_file, 'rb') as f:
                tcache: TrainCache = pickle.load(f)
                fig, ax = plt.subplots()
                tcache.visualize((fig, ax))
                image_file = os.path.join(plot_dir, fname + '.png')
                fig.savefig(image_file)
                print('saved to {}'.format(image_file))
