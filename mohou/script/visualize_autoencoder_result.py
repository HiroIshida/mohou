import argparse

from mohou.default import auto_detect_autoencoder_type
from mohou.model.autoencoder import VariationalAutoEncoder
from mohou.script_utils import (
    visualize_image_reconstruction,
    visualize_variational_autoencoder,
)
from mohou.setting import setting
from mohou.trainer import TrainCache
from mohou.types import MultiEpisodeChunk

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-pn", type=str, default=setting.primary_project_name, help="project name")
    parser.add_argument("-n", type=int, default=5, help="number of visualization")
    args = parser.parse_args()
    project_name = args.pn
    n_vis = args.n

    ae_type = auto_detect_autoencoder_type(project_name)
    model = TrainCache.load(project_name, ae_type).best_model
    assert model is not None

    chunk = MultiEpisodeChunk.load(project_name)
    visualize_image_reconstruction(project_name, chunk, model, n_vis)

    if ae_type == VariationalAutoEncoder:
        visualize_variational_autoencoder(project_name)
