#!/usr/bin/env python3
import pickle
import subprocess
from hashlib import md5
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from mohou.default import create_default_encoding_rule, create_default_propagator
from mohou.model import LSTM, VariationalAutoEncoder
from mohou.propagator import Propagator
from mohou.trainer import TrainCache
from mohou.types import AngleVector, EpisodeBundle


def test_episode_bundle_loading(project_path: Path):
    bundle = EpisodeBundle.load(project_path)
    bundle_hash = md5(pickle.dumps(bundle)).hexdigest()
    print(bundle_hash)
    assert bundle_hash == "614ba6c2c1a07c9f5f96f1f1f25fbed4", "hash value of bundle does not match"
    # test bundle is ok (under construction)
    bundle.plot_vector_histories(AngleVector, project_path)
    # assert (project_path / "seq-AngleVector.png").exists()

    episode = bundle[0]
    gif_path = project_path / "hoge.gif"
    episode.save_debug_gif(gif_path)
    assert gif_path.exists()
    print("bundle loading succeeded")


def test_encoding_rule(project_path: Path):
    encoding_rule = create_default_encoding_rule(project_path)
    bundle = EpisodeBundle.load(project_path)
    arr_list = encoding_rule.apply_to_episode_bundle(bundle)
    sum_value = sum([np.sum(arr) for arr in arr_list])
    print(sum_value)
    # here we do not use md5sum because the encoded value may change slightly depneding
    # on python version
    assert abs(sum_value - (-2799.7209571566077)) < 1e-4, "arr list sum does not match"


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
    # The bunld data comes from https://drive.google.com/drive/folders/1k09pmqxyU5vnEnfaSzTkk-ZL-N8XuoH6
    # The model data comes from pybullet_reaching_RGB demo at v0.3.10
    # https://drive.google.com/drive/u/0/folders/1RQU76D5YpKuQ81AZfPMU1YlgIdNrliyt

    with TemporaryDirectory() as td:
        subprocess.run(
            "cd {} && git clone https://github.com/HiroIshida/mohou_data.git --depth 1".format(
                Path(td)
            ),
            shell=True,
        )
        pp = Path(td) / "mohou_data" / "regression_test"

        # test main
        test_episode_bundle_loading(pp)
        test_encoding_rule(pp)
        test_trained_model_replay(pp)
