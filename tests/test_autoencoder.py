import pytest
from typing import Type

from mohou.model import AutoEncoder, AutoEncoderConfig
from mohou.types import DepthImage, RGBImage, RGBDImage, PrimitiveImageBase


@pytest.mark.parametrize('T', [RGBImage, RGBDImage, DepthImage])
def test_autoencoder(T: Type[PrimitiveImageBase]):
    config = AutoEncoderConfig(T)
    model = AutoEncoder(config)

    # test forward function
    img: PrimitiveImageBase = T.dummy_from_shape(config.input_shape)
    tensor_img_reconstructed = model.forward(img.to_tensor().unsqueeze(dim=0))
    T.from_tensor(tensor_img_reconstructed.squeeze(dim=0))

    # test embedder
    model.get_embedder()
