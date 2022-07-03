import numpy as np
import pytest
from test_file import tmp_project_name  # noqa

from mohou.file import get_project_path, remove_project
from mohou.model import LSTM, LSTMConfig
from mohou.model.common import FloatLossDict
from mohou.trainer import TrainCache


def test_traincache_load(tmp_project_name):  # noqa
    remove_project(tmp_project_name)
    get_project_path(tmp_project_name)

    conf = LSTMConfig(7, 7, 777, 2)
    model = LSTM(conf)

    tcache = TrainCache(tmp_project_name)
    tcache.validate_loss_dict_seq = [FloatLossDict({"loss": 1.0})]  # dummy
    tcache.best_model = model
    tcache.dump()

    # test loadnig
    TrainCache.load(tmp_project_name, LSTM)
    TrainCache.load(tmp_project_name, LSTM, conf)

    with pytest.raises(FileNotFoundError):
        wrong_conf = LSTMConfig(6, 6, 666, 2)
        TrainCache.load(tmp_project_name, LSTM, wrong_conf)


def test_traincache_load_best_one(tmp_project_name):  # noqa
    remove_project(tmp_project_name)
    get_project_path(tmp_project_name)

    conf = LSTMConfig(7, 7, 777, 2)
    model = LSTM(conf)

    def dump_tcache(loss_val):
        tcache = TrainCache(tmp_project_name)
        tcache.validate_loss_dict_seq = [FloatLossDict({"loss": loss_val})]
        tcache.best_model = model
        tcache.dump()

    for loss_value in np.linspace(3, 10, 5):
        dump_tcache(loss_value)

    tcache = TrainCache.load(tmp_project_name, LSTM)
    # must pick up the one with lowest loss
    assert tcache.validate_loss_dict_seq[-1].total() == 3.0
