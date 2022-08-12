import subprocess
import tempfile
from pathlib import Path

import gdown


def get_asset_cache_path() -> Path:
    return Path("~/.mohou/asset").expanduser()


def get_panda_urdf_path() -> Path:
    """get franka_panda's urdf path
    If not file exists, model file will be downloaded.
    NOTE that, the urdf model is modified.
    """
    asset_cache_path = get_asset_cache_path()
    urdf_base_path = asset_cache_path / "franka_panda"
    urdf_file_path = urdf_base_path / "panda.urdf"

    if not urdf_file_path.exists():
        with tempfile.TemporaryDirectory() as td:
            urdf_base_path.mkdir(exist_ok=True, parents=True)

            temp_tarball_path = Path(td) / "tmp.tar"
            url = "https://drive.google.com/uc?id=1gZwXO1jDISZOVIFkC6exFAAk8JLeKco5"
            gdown.download(url, str(temp_tarball_path))
            assert temp_tarball_path.exists()
            subprocess.run(
                "cd {} && tar xf {}".format(urdf_base_path, temp_tarball_path), shell=True
            )
    assert urdf_file_path.exists()
    return urdf_file_path
