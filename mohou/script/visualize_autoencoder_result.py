import argparse
from pathlib import Path

from mohou.default import auto_detect_autoencoder_type
from mohou.file import get_project_path
from mohou.model import Chimera, VariationalAutoEncoder
from mohou.script_utils import (
    visualize_image_reconstruction,
    visualize_variational_autoencoder,
)
from mohou.setting import setting
from mohou.trainer import TrainCache
from mohou.types import EpisodeBundle

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-pn", type=str, default=setting.primary_project_name, help="project name")
    parser.add_argument("-pp", type=str, help="project path. preferred over pn.")
    parser.add_argument("-n", type=int, default=5, help="number of visualization")
    parser.add_argument("--chimera", action="store_true", help="use chimera")
    args = parser.parse_args()
    project_name: str = args.pn
    project_path_str: str = args.pp
    n_vis: int = args.n

    if project_path_str is None:
        project_path = get_project_path(project_name)
    else:
        project_path = Path(project_path_str)

    bundle = EpisodeBundle.load(project_path)

    if args.chimera:
        chimera = TrainCache.load(project_path, Chimera).best_model
        assert chimera is not None
        visualize_image_reconstruction(project_path, bundle, chimera.ae, n_vis, prefix="chimera")
    else:
        ae_type = auto_detect_autoencoder_type(project_path)
        model = TrainCache.load(project_path, ae_type).best_model
        assert model is not None
        visualize_image_reconstruction(project_path, bundle, model, n_vis)

        if ae_type == VariationalAutoEncoder:  # TODO(HiroIshida): enable this also for chimera
            visualize_variational_autoencoder(project_path)
