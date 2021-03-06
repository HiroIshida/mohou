import numpy as np
from test_types import image_av_bundle  # noqa

from mohou.encoder import VectorPCAEncoder
from mohou.types import AngleVector, EpisodeBundle


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
