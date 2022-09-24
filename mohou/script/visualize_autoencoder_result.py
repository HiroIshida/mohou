import argparse
from pathlib import Path

from mohou.encoder import ImageEncoder
from mohou.file import get_project_path
from mohou.script_utils import visualize_image_reconstruction
from mohou.setting import setting
from mohou.types import EpisodeBundle

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-pn", type=str, default=setting.primary_project_name, help="project name")
    parser.add_argument("-pp", type=str, help="project path. preferred over pn.")
    parser.add_argument("-n", type=int, default=5, help="number of visualization")
    args = parser.parse_args()
    project_name: str = args.pn
    project_path_str: str = args.pp
    n_vis: int = args.n

    if project_path_str is None:
        project_path = get_project_path(project_name)
    else:
        project_path = Path(project_path_str)

    bundle = EpisodeBundle.load(project_path)

    image_encoder = ImageEncoder.create_default(project_path)
    model = image_encoder.model
    visualize_image_reconstruction(project_path, bundle, model, n_vis)
