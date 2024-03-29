from typing import List

from mohou.model.autoencoder import AutoEncoderConfig
from mohou.model.common import ModelConfigBase
from mohou.model.experimental import MarkoveModelConfig
from mohou.model.lstm import LSTMConfig
from mohou.types import RGBImage


def test_config_dump():
    configs: List[ModelConfigBase] = []
    configs.append(AutoEncoderConfig(RGBImage))
    configs.append(LSTMConfig(10))
    configs.append(MarkoveModelConfig(10, 10))

    for config in configs:
        config.to_dict()
