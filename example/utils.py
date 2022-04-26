from typing import Optional, Type
from mohou.model import AutoEncoderBase, AutoEncoder, VariationalAutoEncoder
from mohou.trainer import TrainCache


def auto_detect_autoencoder_type(project_name: str) -> Type[AutoEncoderBase]:
    # TODO(HiroIshida) maybe move to script_utils ??
    t: Optional[Type[AutoEncoderBase]] = None

    detect_count = 0
    try:
        TrainCache.load(project_name, AutoEncoder)
        t = AutoEncoder
        detect_count += 1
    except Exception:
        pass

    try:
        TrainCache.load(project_name, VariationalAutoEncoder)
        t = VariationalAutoEncoder
        detect_count += 1
    except Exception:
        pass

    assert detect_count == 1
    assert t is not None  # redundant but for mypy check
    return t
