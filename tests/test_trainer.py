import numpy as np
import pytest
from test_file import tmp_project_name  # noqa

from mohou.file import get_project_path, remove_project
from mohou.model import LSTM, LSTMConfig
from mohou.model.common import FloatLossDict
from mohou.trainer import TrainCache


def test_fld_npz_dict_conversion():
    flds = []
    for _ in range(10):
        flds.append({"loss": abs(np.random.randn())})
        print(flds[-1])
    npz_dict = TrainCache.dump_flds_as_npz_dict(flds)
    flds_again = TrainCache.load_flds_from_npz_dict(npz_dict)
    assert flds == flds_again


def dump_train_cache(conf, loss_value, project_name):
    model = LSTM(conf)
    tcache = TrainCache.from_model(model)
    fld = FloatLossDict({"loss": loss_value})
    tcache.update_and_save(model, fld, fld, project_name)


def test_traincache_load_all(tmp_project_name):  # noqa
    get_project_path(tmp_project_name)

    conf = LSTMConfig(7, 7, 777, 2)  # whatever
    for _ in range(10):
        dump_train_cache(conf, np.random.rand(), tmp_project_name)

    conf2 = LSTMConfig(3, 3, 3, 2)  # whatever
    for _ in range(10):
        dump_train_cache(conf2, np.random.rand(), tmp_project_name)

    assert len(TrainCache.load_all(tmp_project_name)) == 20
    assert len(TrainCache.load_all(tmp_project_name, LSTM)) == 20
    assert len(TrainCache.load_all(tmp_project_name, LSTM, conf)) == 10
    assert len(TrainCache.load_all(tmp_project_name, LSTM, conf2)) == 10

    remove_project(tmp_project_name)


def test_traincache_load(tmp_project_name):  # noqa
    get_project_path(tmp_project_name)

    conf = LSTMConfig(7, 7, 777, 2)
    dump_train_cache(conf, 1.0, tmp_project_name)

    # test loading
    TrainCache.load(tmp_project_name, LSTM)
    TrainCache.load(tmp_project_name, LSTM, conf)

    with pytest.raises(FileNotFoundError):
        wrong_conf = LSTMConfig(6, 6, 666, 2)
        TrainCache.load(tmp_project_name, LSTM, wrong_conf)

    remove_project(tmp_project_name)


def test_traincache_load_best_one(tmp_project_name):  # noqa
    get_project_path(tmp_project_name)

    conf = LSTMConfig(7, 7, 777, 2)

    for loss_value in np.linspace(3, 10, 5):
        dump_train_cache(conf, loss_value, tmp_project_name)

    tcache = TrainCache.load(tmp_project_name, LSTM)
    # must pick up the one with lowest loss
    assert tcache.validate_loss_dict_seq[-1].total() == 3.0

    remove_project(tmp_project_name)
