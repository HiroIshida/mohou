import torch

from mohou.model import LSTM, LSTMConfig


def test_lstm():
    n_sample = 10
    n_seq_len = 100
    n_dim_with_flag = 7
    n_dim_static_context = 4

    config = LSTMConfig(n_dim_with_flag, n_static_context=n_dim_static_context)
    model: LSTM = LSTM(config)

    state_sample = torch.randn(n_sample, n_seq_len, n_dim_with_flag).float()
    ti_inputs = torch.randn(n_sample, n_dim_static_context).float()

    # test forward
    state_prop, _ = model.forward(state_sample, ti_inputs)
    assert state_prop.shape == (n_sample, n_seq_len, n_dim_with_flag)

    # test loss
    weight_sample = torch.rand(n_sample, n_seq_len)
    sample = (state_sample, ti_inputs, weight_sample)
    loss_dict = model.loss(sample)
    assert "prediction" in loss_dict
    assert loss_dict.total() == loss_dict["prediction"]
