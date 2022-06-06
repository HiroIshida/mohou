import numpy as np

from mohou.types import AngleVector, MultiEpisodeChunk
from mohou.encoder import VectorPCAEncoder

from test_types import image_av_chunk  # noqa


def test_pca_embedder(image_av_chunk):  # noqa
    chunk: MultiEpisodeChunk = image_av_chunk
    pca_emb = VectorPCAEncoder.from_chunk(chunk, AngleVector, 3)
    assert pca_emb.input_shape == (10,)
    assert pca_emb.output_size == 3
    feature = pca_emb.forward(AngleVector(np.random.randn(10)))
    pca_emb.backward(feature)

    # must be identical reconsturction (because of the same dim)
    pca_emb = VectorPCAEncoder.from_chunk(chunk, AngleVector, 10)
    inp = AngleVector(np.random.randn(10))
    feature = pca_emb.forward(inp)
    reconstructed = pca_emb.backward(feature)
    np.testing.assert_almost_equal(reconstructed.numpy(), inp.numpy())
