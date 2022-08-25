#!/usr/bin/env python3

from pathlib import Path
from tempfile import TemporaryDirectory

import gdown

from mohou.default import create_default_propagator
from mohou.model import LSTM, VariationalAutoEncoder
from mohou.propagator import Propagator
from mohou.trainer import TrainCache
from mohou.types import AngleVector, EpisodeBundle


def test_episode_bundle_loading(project_path: Path):
    bundle = EpisodeBundle.load(project_path)

    # test bundle is ok (under construction)
    bundle.plot_vector_histories(AngleVector, project_path)
    assert (project_path / "seq-AngleVector.png").exists()

    episode = bundle[0]
    gif_path = project_path / "hoge.gif"
    episode.save_debug_gif(gif_path)
    assert gif_path.exists()
    print("bundle loading succeeded")


def test_trained_model_replay(project_path: Path):
    # check at least model loading success
    TrainCache.load(project_path, LSTM)
    TrainCache.load(project_path, VariationalAutoEncoder)

    # check propagator can be constructed using LSTM and autoencoder
    prop = create_default_propagator(project_path, Propagator)
    bundle = EpisodeBundle.load(project_path).get_untouch_bundle()
    episode = bundle[0]
    for i in range(10):
        prop.feed(episode[i])
    prop.predict(5)
    print("model loading succeeded")


if __name__ == "__main__":
    # The bundle data comes from pybullet_reaching_RGB demo at v0.3.10
    # https://drive.google.com/drive/u/0/folders/1RQU76D5YpKuQ81AZfPMU1YlgIdNrliyt

    with TemporaryDirectory() as td:
        # download data
        pp = Path(td)
        pp.mkdir(exist_ok=True)
        bundle_url = "https://drive.google.com/uc?id=1J05WWSeDEzpjx1Z5xbWDT2Dc9J0gng_h"
        bundle_path = pp / "EpisodeBundle.tar"
        gdown.download(bundle_url, str(bundle_path), quiet=False)

        model_path = pp / "models"
        model_url = "https://drive.google.com/drive/folders/1ns0xoggajMUjjMNyBd4_k9VDodquRFYi"
        gdown.download_folder(model_url, output=str(model_path))

        assert len(list(model_path.iterdir())) > 0, "likely that donwload failed"

        # test main
        test_episode_bundle_loading(pp)
        test_trained_model_replay(pp)
