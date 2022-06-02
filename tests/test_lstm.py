import torch

from mohou.model import LSTM, LSTMConfig


def test_lstm():
    n_sample = 10
    n_seq_len = 100
    n_dim_with_flag = 7

    config = LSTMConfig(n_dim_with_flag)
    model: LSTM = LSTM(config)

    state_sample = torch.randn(n_sample, n_seq_len, n_dim_with_flag).float()

    # test forward
    state_prop = model.forward(state_sample)
    assert state_prop.shape == (n_sample, n_seq_len, n_dim_with_flag)

    # test loss
    sample = (state_sample, torch.rand(n_sample, n_seq_len))
    loss_dict = model.loss(sample)
    assert 'prediction' in loss_dict
    assert loss_dict.total() == loss_dict['prediction']
