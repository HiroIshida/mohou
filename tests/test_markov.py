import torch

from mohou.model import ControlModel, MarkoveModelConfig


def test_control_equation_model():
    n_obs_dim = 10
    n_ctrl_dim = 5
    n_batch = 10
    config = MarkoveModelConfig(n_input=n_obs_dim + n_ctrl_dim, n_output=n_obs_dim)

    model: ControlModel = ControlModel(config)

    inp_ctrl_sample = torch.randn(n_batch, n_ctrl_dim)
    inp_obs_sample = torch.randn(n_batch, n_obs_dim)
    out_obs_sample = torch.randn(n_batch, n_obs_dim)

    loss_dict = model.loss((inp_ctrl_sample, inp_obs_sample, out_obs_sample))
    assert "prediction" in loss_dict
    assert loss_dict.total() == loss_dict["prediction"]
