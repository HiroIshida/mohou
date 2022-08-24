#!/usr/bin/env python3

from pathlib import Path
from tempfile import TemporaryDirectory

import gdown

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


if __name__ == "__main__":
    # The bundle data comes from pybullet_reaching_RGB demo at v0.3.10
    # https://drive.google.com/drive/u/0/folders/1RQU76D5YpKuQ81AZfPMU1YlgIdNrliyt

    with TemporaryDirectory() as td:
        pp = Path(td)
        pp.mkdir(exist_ok=True)
        bundle_url = "https://drive.google.com/uc?id=1J05WWSeDEzpjx1Z5xbWDT2Dc9J0gng_h"
        bundle_path = pp / "EpisodeBundle.tar"
        gdown.download(bundle_url, str(bundle_path), quiet=False)
        test_episode_bundle_loading(pp)

        # TODO: add test_model_loading
