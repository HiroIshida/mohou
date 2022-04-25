import argparse
from mohou.script_utils import visualize_image_reconstruction

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-pn', type=str, default='kuka_reaching', help='project name')
    parser.add_argument('-n', type=int, default=5, help='number of visualization')
    args = parser.parse_args()
    project_name = args.pn
    n_vis = args.n
    visualize_image_reconstruction(project_name, n_vis)
