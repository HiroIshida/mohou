#!/usr/bin/env python3
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from mohou.default import create_default_encoding_rule, create_default_propagator
from mohou.propagator import Propagator
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


def test_propagator(project_path: Path):
    prop: Propagator = create_default_propagator(project_path, Propagator)
    bundle = EpisodeBundle.load(project_path)
    episode = bundle[0]

    for i in range(40):
        prop.feed(episode[i])
    edict_seq = prop.predict(40)
    episode = EpisodeData.from_edict_list(edict_seq, check_terminate_flag=False)

    episode.get_sequence_by_type(AngleVector)
    episode.get_sequence_by_type(RGBImage)
    episode.get_sequence_by_type(TerminateFlag)

    sum_value = 0.0
    for elem_type in [AngleVector, RGBImage, TerminateFlag]:
        seq = episode.get_sequence_by_type(elem_type)  # type: ignore
        for elem in seq:
            sum_value += np.sum(elem.numpy())
    print(sum_value)
    assert abs(sum_value - (1427638191.4711995)) < 1e-5, "sum does not match"


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
        test_propagator(pp)
