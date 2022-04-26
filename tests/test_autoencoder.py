import pytest
from typing import Type

import numpy as np

from mohou.model import AutoEncoder, VariationalAutoEncoder, AutoEncoderConfig
from mohou.types import DepthImage, RGBImage, RGBDImage, PrimitiveImageBase


@pytest.mark.parametrize('S', [28, 112, 224])
@pytest.mark.parametrize('T', [RGBImage, RGBDImage, DepthImage])
@pytest.mark.parametrize('M', [AutoEncoder, VariationalAutoEncoder])
def test_autoencoder(S: int, T: Type[PrimitiveImageBase], M: Type):
    config = AutoEncoderConfig(T, n_pixel=S)
    model: AutoEncoder = M(config)  # TODO(HiroIShida) fix this
    img: PrimitiveImageBase = T.dummy_from_shape((config.n_pixel, config.n_pixel))

    # test forward function
    sample = img.to_tensor().unsqueeze(dim=0)
    tensor_img_reconstructed = model.forward(sample)
    T.from_tensor(tensor_img_reconstructed.squeeze(dim=0))

    # test image size assertion
    with pytest.raises(AssertionError):
        img_strange: PrimitiveImageBase = T.dummy_from_shape((10, 10))
        model.forward(img_strange.to_tensor().unsqueeze(dim=0))

    # test embedder (This also test ImageEmbedder)
    embedder = model.get_embedder()
    feature_vec: np.ndarray = embedder.forward(img)
    assert list(feature_vec.shape) == [config.n_bottleneck]
    img2 = embedder.backward(feature_vec)
    assert type(img) == type(img2)
    assert img.channel() == img2.channel()
    assert img.shape == img2.shape

    loss_dict = model.loss(sample)
    if M == AutoEncoder:
        assert 'reconstruction' in loss_dict
        loss_dict.total() == loss_dict['reconstruction']
    elif M == VariationalAutoEncoder:
        assert 'reconstruction' in loss_dict
        assert 'kld' in loss_dict
        loss_dict.total() == loss_dict['reconstruction'] + loss_dict['kld']
