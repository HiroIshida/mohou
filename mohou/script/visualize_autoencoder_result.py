import argparse

from mohou.default import auto_detect_autoencoder_type
from mohou.model.autoencoder import VariationalAutoEncoder
from mohou.script_utils import (
    visualize_image_reconstruction,
    visualize_variational_autoencoder,
)
from mohou.setting import setting

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-pn", type=str, default=setting.primary_project_name, help="project name")
    parser.add_argument("-n", type=int, default=5, help="number of visualization")
    args = parser.parse_args()
    project_name = args.pn
    n_vis = args.n
    ae_type = auto_detect_autoencoder_type(project_name)
    visualize_image_reconstruction(project_name, n_vis, ae_type=ae_type)

    if ae_type == VariationalAutoEncoder:
        visualize_variational_autoencoder(project_name)
