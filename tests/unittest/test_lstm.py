import copy

import torch
from test_encoding_rule import create_encoding_rule_for_image_av_bundle
from test_types import image_av_bundle  # noqa

from mohou.model import LSTM, PBLSTM, LSTMConfig, PBLSTMConfig


def test_lstm(image_av_bundle):  # noqa
    rule = create_encoding_rule_for_image_av_bundle(image_av_bundle)

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

    episode_indices = torch.randint(0, 1, (n_sample,))
    state_sample = torch.randn(n_sample, n_seq_len, n_dim_with_flag).float()
    context_sample = torch.randn(n_sample, n_dim_static_context).float()

    # test forward
    state_prop, _ = model.forward(state_sample, context_sample)
    assert state_prop.shape == (n_sample, n_seq_len, n_dim_with_flag)

    # test normal loss
    sample = (episode_indices, state_sample, context_sample)
    loss_dict = model.loss(sample)
    assert len(loss_dict.keys()) == 1

    # test type_wise_loss
    # This test is bit hacky. The reason why I copied the model is to match the
    # neural network parameters of model1 and model2 to be equal.
    # And, there is assumtion that type_wise_loss has no effect in creation of
    # the lstm model
    config.type_wise_loss = True
    model2 = copy.deepcopy(model)
    model2.config = config
    loss_dict_detailed = model2.loss(sample)
    assert len(loss_dict_detailed.keys()) == len(rule.keys())
    error = loss_dict_detailed.to_float_lossdict().total() - loss_dict.to_float_lossdict().total()
    assert abs(error) < 1e-6

    loss_dict.total().backward()
    loss_dict_detailed.total().backward()

    for param1, param2 in zip(model.parameters(), model2.parameters()):
        assert param1.grad is not None
        assert param2.grad is not None
        assert torch.allclose(param1.grad, param2.grad)


def test_lstm_with_window(image_av_bundle):  # noqa

    rule = create_encoding_rule_for_image_av_bundle(image_av_bundle)

    n_sample = 10
    n_dim_with_flag = rule.dimension
    n_dim_static_context = 4
    window_size = 10

    config = LSTMConfig(
        n_dim_with_flag,
        n_static_context=n_dim_static_context,
        type_bound_table=rule.type_bound_table,
        window_size=window_size,
    )

    model: LSTM = LSTM(config)
    for n_seq_len in [5, window_size, 15]:
        sample = torch.randn((n_sample, n_seq_len, n_dim_with_flag))
        context_sample = torch.randn(n_sample, n_dim_static_context).float()
        out, _ = model.forward(sample, context_sample)
        assert tuple(out.shape) == (n_sample, min(n_seq_len, window_size), n_dim_with_flag)


def test_pblstm(image_av_bundle):  # noqa

    rule = create_encoding_rule_for_image_av_bundle(image_av_bundle)

    n_sample = 10
    n_seq_len = 100
    n_pb = 20
    n_pb_dim = 2
    n_state_dim = rule.dimension

    config = PBLSTMConfig(n_state_with_flag=n_state_dim, n_pb=n_pb, n_pb_dim=n_pb_dim)
    model = PBLSTM(config)

    state_sample = torch.randn(n_sample, n_seq_len, n_state_dim).float()
    context_sample = torch.randn(n_sample, 0).float()
    pb_sample = torch.randn(n_sample, n_pb_dim)

    # test forward
    state_prop, _ = model.forward(state_sample, pb_sample, context_sample)
    assert state_prop.shape == (n_sample, n_seq_len, n_state_dim)

    # test normal loss
    indices = torch.randint(0, n_pb, (n_sample,))
    sample = (indices, state_sample, context_sample)
    loss_dict = model.loss(sample)
    assert len(loss_dict.keys()) == 1
    assert loss_dict.total().item() > 0
