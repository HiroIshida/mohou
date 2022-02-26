import pytest
from typing import Type

import numpy as np

from mohou.model import AutoEncoder, AutoEncoderConfig
from mohou.types import DepthImage, RGBImage, RGBDImage, PrimitiveImageBase


@pytest.mark.parametrize('T', [RGBImage, RGBDImage, DepthImage])
def test_autoencoder(T: Type[PrimitiveImageBase]):
    config = AutoEncoderConfig(T)
    model: AutoEncoder = AutoEncoder(config)
    img: PrimitiveImageBase = T.dummy_from_shape(config.input_shape)

    # test forward function
    tensor_img_reconstructed = model.forward(img.to_tensor().unsqueeze(dim=0))
    T.from_tensor(tensor_img_reconstructed.squeeze(dim=0))

    # test embedder (This also test ImageEmbedder)
    embedder = model.get_embedder()
    feature_vec: np.ndarray = embedder.forward(img)
    assert list(feature_vec.shape) == [config.n_bottleneck]
    img2 = embedder.backward(feature_vec)
    assert type(img) == type(img2)
    assert img.channel() == img2.channel()
    assert img.shape == img2.shape
