#!/usr/bin/env python3
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from mohou.default import create_default_encoding_rule, create_default_propagator
from mohou.model import LSTM, VariationalAutoEncoder
from mohou.propagator import Propagator
from mohou.trainer import TrainCache
from mohou.types import AngleVector, EpisodeBundle, EpisodeData, RGBImage, TerminateFlag


def test_episode_bundle_loading(project_path: Path):
    bundle = EpisodeBundle.load(project_path)

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


def test_ae_model(project_path: Path):
    ae_cache = TrainCache.load(project_path, VariationalAutoEncoder)
    vae = ae_cache.best_model

    create_default_encoding_rule(project_path)
    bundle = EpisodeBundle.load(project_path)
    episode = bundle[0]
    image_seq = episode.get_sequence_by_type(RGBImage)

    # do not put many image. float rooted numerical error becomes large
    arr = torch.stack([image.to_tensor() for image in image_seq[0:3]])
    ret = vae.forward(arr)
    sum_value = torch.sum(ret).item()
    print(sum_value)
    assert abs(sum_value - (415137.75)) < 1e-4, "ae sum does not match"


def test_lstm_model(project_path: Path):
    lstm_cache = TrainCache.load(project_path, LSTM)
    lstm = lstm_cache.best_model
    lstm.forward

    encoding_rule = create_default_encoding_rule(project_path)
    bundle = EpisodeBundle.load(project_path)
    episode = bundle[0]
    arr = encoding_rule.apply_to_episode_data(episode)

    dummy_sample = torch.from_numpy(arr).float().unsqueeze(dim=0)
    n_batch, n_seqlen, n_dim = dummy_sample.shape
    dummy_context = torch.randn(n_batch, 0)
    val, _ = lstm.forward(dummy_sample, dummy_context)
    sum_value = torch.sum(val).item()
    print(sum_value)
    assert abs(sum_value - (31.875625610351562)) < 1e-4, "lstm sum does not match"


def test_propagator(project_path: Path):
    prop: Propagator = create_default_propagator(project_path, Propagator)
    bundle = EpisodeBundle.load(project_path)
    episode = bundle[0]

    for i in range(40):
        prop.feed(episode[i])
    edict_seq = prop.predict(2)
    episode = EpisodeData.from_edict_list(edict_seq, check_terminate_flag=False)

    episode.get_sequence_by_type(AngleVector)
    episode.get_sequence_by_type(RGBImage)
    episode.get_sequence_by_type(TerminateFlag)

    sum_value = 0.0
    for elem_type in [AngleVector, TerminateFlag]:
        # not including RGB because its type is uint8
        seq = episode.get_sequence_by_type(elem_type)  # type: ignore
        for elem in seq:
            sum_value += np.sum(elem.numpy())
    print(sum_value)
    assert abs(sum_value - (0.2111293077468872)) < 1e-6, "sum does not match"


if __name__ == "__main__":

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
        test_ae_model(pp)
        test_lstm_model(pp)
        test_propagator(pp)
