#!/usr/bin/env python3

from pathlib import Path
from tempfile import TemporaryDirectory

import gdown

from mohou.types import AngleVector, EpisodeBundle


def test_episode_bundle_loading(project_path: Path):
    bundle = EpisodeBundle.load(Path)

    # test bundle is ok (under construction)
    bundle.plot_vector_histories(AngleVector, project_path)
    assert (project_path / "seq-AngleVector.png").exists()

    episode = bundle[0]
    gif_path = project_path / "hoge.gif"
    episode.save_debug_gif(gif_path)
    assert gif_path.exists()


if __name__ == "__main__":
    with TemporaryDirectory() as td:
        pp = Path(td)
        pp.mkdir(exist_ok=True)

        bundle_url = "https://drive.google.com/uc?id=1J05WWSeDEzpjx1Z5xbWDT2Dc9J0gng_h"
        bundle_path = pp / "EpisodeBundle.tar"
        gdown.download(bundle_url, bundle_path, quiet=False)

    test_episode_bundle_loading(pp)
