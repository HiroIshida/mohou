import argparse
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

from mohou.file import get_project_path
from mohou.model.lstm import PBLSTM
from mohou.trainer import TrainCache
from mohou.types import EpisodeBundle

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-pn", type=str, help="project name")
    parser.add_argument("-n_pb_dim", type=int, help="pb dimension")

    args = parser.parse_args()
    project_name: Optional[str] = args.pn
    n_pb_dim: Optional[str] = args.n_pb_dim

    project_path = get_project_path(project_name)

    tcache_search_kwargs = {}
    if n_pb_dim is not None:
        tcache_search_kwargs["n_pb_dim"] = n_pb_dim

    tcache = TrainCache.load(project_path, PBLSTM, **tcache_search_kwargs)
    model = tcache.best_model
    assert model.config.n_pb_dim > 1, "(future plan) implement line plot?"

    pb_list = [pb_tensor.detach().numpy() for pb_tensor in model.parametric_bias_list]
    pb_arr = np.array(pb_list)

    pca = PCA(2)
    pca.fit(pb_arr)
    pb_reduced_arr = pca.transform(pb_arr)

    bundle = EpisodeBundle.load(project_path)
    fig, ax = plt.subplots()
    ax.scatter(pb_reduced_arr[:, 0], pb_reduced_arr[:, 1])
    for idx_episode, pb in enumerate(pb_reduced_arr):
        episode = bundle[idx_episode]
        print(idx_episode)
        print(episode.metadata)
        ax.annotate(str(idx_episode), (pb[0], pb[1]))

    plt.show()
