from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class Setting:
    project_dir_path_list: List[Path]
    primary_project_name: Optional[str]
    n_untouch_episode: int

    @classmethod
    def construct(cls) -> "Setting":

        # define default settings
        default_setting: Dict[str, Any] = {
            "project_dir_path_list": [Path("~/.mohou")],
            "primary_project_name": None,
            "n_untouch_episode": 5,
        }

        # overwrite default setting by loading yaml
        setting_path = Path("~/.mohou").expanduser() / "setting.yaml"
        if not setting_path.exists():
            return cls(**default_setting)  # type: ignore

        with setting_path.open(mode="r") as f:
            setting_load = yaml.safe_load(f)

        invalid_keys = set(setting_load.keys()).difference(default_setting.keys())
        message = "invalid keys {} in setting.yaml".format(invalid_keys)
        assert len(invalid_keys) == 0, message

        if "project_dir_path_list" in setting_load:
            default_setting["project_dir_path_list"] = [
                Path(v).expanduser() for v in setting_load["project_dir_path_list"]
            ]
        if "primary_project_name" in setting_load:
            default_setting["primary_project_name"] = str(setting_load["primary_project_name"])
        if "n_untouch_episode" in setting_load:
            default_setting["n_untouch_episode"] = int(setting_load["n_untouch_episode"])
        print(default_setting)
        return cls(**default_setting)  # type: ignore


setting = Setting.construct()
