import torch
from test_types import image_av_bundle  # noqa

from mohou.model import LSTM, LSTMConfig
from mohou.model.third_party.variational_lstm import VariationalLSTM, WeightDrop


def test_variational_lstm_train_eval_mode():
    config = LSTMConfig(3, variational=True)
    lstm = LSTM(config)
    vlstm: VariationalLSTM = lstm.lstm_layer  # type: ignore

    lstm.train()
    assert lstm.training
    assert vlstm.training
    for rnn in vlstm.lstms:
        assert isinstance(rnn, WeightDrop)
        assert rnn.training
    assert vlstm.lockdrop_inp.training
    assert vlstm.lockdrop_out.training

    lstm.eval()
    assert not lstm.training
    assert not vlstm.training
    for rnn in vlstm.lstms:
        assert not rnn.training
    assert not vlstm.lockdrop_inp.training
    assert not vlstm.lockdrop_out.training


def test_variational_lstm():
    sample = (torch.zeros((5, 10, 3)), torch.zeros((5, 0)))

    config = LSTMConfig(3, variational=True)
    lstm = LSTM(config)

    # in the evaluation mode, dropout mask is fixed
    lstm.eval()
    out1, _ = lstm.forward(*sample)
    out2, _ = lstm.forward(*sample)
    assert torch.norm(out1 - out2).item() < 1e-8

    # in the evaluation mode, dropout mask is not fixed
    lstm.train()
    out1, _ = lstm.forward(*sample)
    out2, _ = lstm.forward(*sample)
    assert not torch.norm(out1 - out2).item() < 1e-8
