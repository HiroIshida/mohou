import argparse
from typing import Type

from mohou.model import AutoEncoderBase, AutoEncoder, VariationalAutoEncoder
from mohou.script_utils import visualize_image_reconstruction

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-pn', type=str, default='kuka_reaching', help='project name')
    parser.add_argument('-n', type=int, default=5, help='number of visualization')
    parser.add_argument('--vae', action='store_true', help='use vae')
    args = parser.parse_args()
    project_name = args.pn
    n_vis = args.n
    use_vae = args.vae
    ae_type: Type[AutoEncoderBase] = VariationalAutoEncoder if use_vae else AutoEncoder  # type: ignore
    visualize_image_reconstruction(project_name, n_vis, ae_type=ae_type)
