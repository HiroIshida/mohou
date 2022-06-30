import pytest
import torch
from test_file import tmp_project_name  # noqa

from mohou.file import get_project_path, remove_project
from mohou.model import LSTM, LSTMConfig
from mohou.model.common import LossDict
from mohou.trainer import TrainCache


def test_traincache_load(tmp_project_name):  # noqa
    remove_project(tmp_project_name)
    get_project_path(tmp_project_name)

    conf = LSTMConfig(7, 7, 777, 2)
    model = LSTM(conf)

    tcache = TrainCache(tmp_project_name)
    tcache.validate_loss_dict_seq = [LossDict({"loss": torch.tensor(1.0)})]  # dummy
    tcache.best_model = model
    tcache.dump()

    # test loadnig
    tcache.load(tmp_project_name, LSTM)
    tcache.load(tmp_project_name, LSTM, conf)

    with pytest.raises(FileNotFoundError):
        wrong_conf = LSTMConfig(6, 6, 666, 2)
        tcache.load(tmp_project_name, LSTM, wrong_conf)
