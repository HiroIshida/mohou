import torch
from test_encoding_rule import create_encoding_rule
from test_types import image_av_chunk  # noqa

from mohou.model import LSTM, LSTMConfig


def test_lstm(image_av_chunk):  # noqa
    rule = create_encoding_rule(image_av_chunk)

    n_sample = 10
    n_seq_len = 100
    n_dim_with_flag = rule.dimension
    n_dim_static_context = 4

    config = LSTMConfig(
        n_dim_with_flag,
        n_static_context=n_dim_static_context,
        type_bound_table=rule.type_bound_table,
    )
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
    assert len(loss_dict.keys()) == 1

    # test type_wise_loss
    config.type_wise_loss = True
    model2: LSTM = LSTM(config)
    loss_dict_detailed = model2.loss(sample)
    assert len(loss_dict_detailed.keys()) == len(rule.keys())
