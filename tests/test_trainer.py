import numpy as np
import pytest
from test_file import tmp_project_name  # noqa

from mohou.file import create_project_dir, get_project_path, remove_project
from mohou.model import LSTM, LSTMConfig
from mohou.model.common import FloatLossDict
from mohou.trainer import TrainCache


def test_fld_npz_dict_conversion():
    table = {
        "loss_a": np.abs(np.random.randn(10)).tolist(),
        "loss_b": np.abs(np.random.randn(10)).tolist(),
    }
    npz_dict = TrainCache.dump_lossseq_table_as_npz_dict(table)
    table_again = TrainCache.load_lossseq_table_from_npz_dict(npz_dict)
    assert table == table_again


def dump_train_cache(conf, loss_value, project_name):
    model = LSTM(conf)
    tcache = TrainCache.from_model(model)
    fld = FloatLossDict({"loss": loss_value})
    tcache.update_and_save(model, fld, fld, project_name)


def test_traincache_load_all(tmp_project_name):  # noqa
    create_project_dir(tmp_project_name)
    tmp_project_path = get_project_path(tmp_project_name)

    conf = LSTMConfig(7, 7, 777, 2)  # whatever
    for _ in range(10):
        dump_train_cache(conf, np.random.rand(), tmp_project_path)

    conf2 = LSTMConfig(3, 3, 3, 2)  # whatever
    for _ in range(10):
        dump_train_cache(conf2, np.random.rand(), tmp_project_path)

    assert len(TrainCache.load_all(tmp_project_path)) == 20
    assert len(TrainCache.load_all(tmp_project_path, LSTM)) == 20
    assert len(TrainCache.load_all(tmp_project_path, LSTM, conf)) == 10
    assert len(TrainCache.load_all(tmp_project_path, LSTM, conf2)) == 10

    remove_project(tmp_project_name)


def test_traincache_load(tmp_project_name):  # noqa
    create_project_dir(tmp_project_name)
    tmp_project_path = get_project_path(tmp_project_name)

    conf = LSTMConfig(7, 7, 777, 2)
    dump_train_cache(conf, 1.0, get_project_path(tmp_project_name))

    # test loading
    TrainCache.load(tmp_project_path, LSTM)
    TrainCache.load(tmp_project_path, LSTM, conf)

    with pytest.raises(FileNotFoundError):
        wrong_conf = LSTMConfig(6, 6, 666, 2)
        TrainCache.load(tmp_project_path, LSTM, wrong_conf)

    remove_project(tmp_project_name)


def test_traincache_load_best_one(tmp_project_name):  # noqa
    create_project_dir(tmp_project_name)
    tmp_project_path = get_project_path(tmp_project_name)

    conf = LSTMConfig(7, 7, 777, 2)

    for loss_value in np.linspace(3, 10, 5):
        dump_train_cache(conf, loss_value, tmp_project_path)

    tcache = TrainCache.load(tmp_project_path, LSTM)
    # must pick up the one with lowest loss
    assert tcache.min_validate_loss == 3.0

    remove_project(tmp_project_name)
