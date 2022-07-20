from dataclasses import MISSING, dataclass
from pathlib import Path
from typing import Optional

import yaml


@dataclass(frozen=True)
class Setting:
    root_path: Path = Path("~/.mohou")
    primary_project_name: Optional[str] = None

    @classmethod
    def construct(cls) -> "Setting":

        default_attributes = {}
        for key, val in cls.__dataclass_fields__.items():  # type: ignore # mypy's bug
            assert val.default != MISSING
            default_attributes[key] = val.default

        setting = None
        setting_path = Path("~/.mohou").expanduser() / "setting.yaml"
        if setting_path.exists():
            with setting_path.open(mode="r") as f:
                setting = yaml.safe_load(f)

        if setting is None:
            return cls()  # use default

        invalid_keys = set(setting.keys()).difference(set(default_attributes.keys()))
        message = "invalid keys {} in setting.yaml".format(invalid_keys)
        assert len(invalid_keys) == 0, message

        for key in default_attributes.keys():
            if key in setting:
                # overwrite
                raw_type = cls.__dataclass_fields__[key].type  # type: ignore # mypy's bug
                if isinstance(raw_type, type):
                    expect_type = raw_type
                else:
                    is_union_t = len(raw_type.__args__) == 2
                    is_optional_t = is_union_t and issubclass(raw_type.__args__[1], type(None))
                    assert is_optional_t
                    expect_type = raw_type.__args__[0]  # the one that is not NoneType
                default_attributes[key] = expect_type(setting[key])  # cast

        return cls(**default_attributes)  # type: ignore

    def __post_init__(self):
        for key, val in self.__dict__.items():
            if isinstance(val, Path):
                self.__dict__[key] = val.expanduser()


setting = Setting.construct()
