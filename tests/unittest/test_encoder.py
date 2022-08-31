import numpy as np
from test_types import image_av_bundle  # noqa

from mohou.encoder import ImageEncoder, VectorIdenticalEncoder, VectorPCAEncoder
from mohou.model import AutoEncoder, AutoEncoderConfig
from mohou.types import AngleVector, EpisodeBundle, RGBImage


def test_image_encoder_serialization():
    config = AutoEncoderConfig(RGBImage, 16, 28)
    model = AutoEncoder(config)  # type: ignore [var-annotated]  # for python 3.6 compat
    encoder = ImageEncoder.from_auto_encoder(model)
    encoder_again = ImageEncoder.from_dict(encoder.to_dict())
    assert encoder == encoder_again


def test_av_identical_encoder_serialization():
    encoder = VectorIdenticalEncoder.create(AngleVector, 7)
    encoder_again = VectorIdenticalEncoder[AngleVector].from_dict(encoder.to_dict())
    assert encoder == encoder_again


def test_pca_encoder(image_av_bundle):  # noqa
    bundle: EpisodeBundle = image_av_bundle
    pca_emb = VectorPCAEncoder.from_bundle(bundle, AngleVector, 3)
    assert pca_emb.input_shape == (10,)
    assert pca_emb.output_size == 3
    feature = pca_emb.forward(AngleVector(np.random.randn(10)))
    pca_emb.backward(feature)

    # must be identical reconsturction (because of the same dim)
    pca_emb = VectorPCAEncoder.from_bundle(bundle, AngleVector, 10)
    inp = AngleVector(np.random.randn(10))
    feature = pca_emb.forward(inp)
    reconstructed = pca_emb.backward(feature)
    assert reconstructed == inp
