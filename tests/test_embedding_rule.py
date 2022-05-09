import pytest
import numpy as np

from itertools import permutations
import torch

from mohou.embedder import ImageEmbedder, IdenticalEmbedder
from mohou.embedding_rule import EmbeddingRule
from mohou.embedding_rule import ElemCovMatchPostProcessor
from mohou.types import AngleVector, RGBImage, RGBDImage, TerminateFlag, VectorBase

from test_types import image_av_chunk # noqa


def test_ElemCovMatchPostProcessor():
    dim1 = 2
    dim2 = 3
    bias = 10
    a = np.random.randn(100000, dim1) + np.ones(2) * bias
    b = np.random.randn(100000, dim2)
    b[:, 0] *= 3
    b[:, 1] *= 2
    b[:, 2] *= 0.5
    c = np.concatenate([a, b], axis=1)
    normalizer = ElemCovMatchPostProcessor.from_feature_seqs(c, [dim1, dim2])
    inp = np.random.randn(5)
    normalized = normalizer.apply(inp)
    denormalized = normalizer.inverse_apply(normalized)
    np.testing.assert_almost_equal(inp, denormalized, decimal=2)

    cstds = normalizer.characteristic_stds
    np.testing.assert_almost_equal(cstds, np.array([1.0, 3.0]), decimal=1)
    scaled_cstds = normalizer.scaled_characteristic_stds
    np.testing.assert_almost_equal(scaled_cstds, np.array([1.0 / 3.0, 1.0]), decimal=2)


def test_embedding_rule(image_av_chunk): # noqa
    chunk = image_av_chunk
    n_image_embed = 5
    n_av_embed = 10
    f1 = ImageEmbedder(
        RGBImage,
        lambda img: torch.zeros(n_image_embed),
        lambda vec: torch.zeros(3, 100, 100),
        (100, 100, 3), n_image_embed)
    f2 = IdenticalEmbedder(AngleVector, n_av_embed)
    f3 = IdenticalEmbedder(TerminateFlag, 1)

    rule = EmbeddingRule.from_embedders([f1, f2, f3], chunk=chunk)
    vector_seq_list = rule.apply_to_multi_episode_chunk(chunk)
    vector_seq = vector_seq_list[0]

    assert vector_seq.shape == (10, n_image_embed + n_av_embed + 1)

    class Dummy(VectorBase):
        pass

    # Check dict insertion oreder is preserved
    # NOTE: from 3.7, order is preserved as lang. spec.
    # NOTE: from 3.6, order is preserved in a cpython implementation
    f3 = IdenticalEmbedder(TerminateFlag, 1)
    f4 = IdenticalEmbedder(Dummy, 2)
    pairs = [(f1, RGBImage), (f2, AngleVector), (f3, TerminateFlag), (f4, Dummy)]
    for pairs_perm in permutations(pairs, 4):
        fs = [p[0] for p in pairs_perm]
        ts = [p[1] for p in pairs_perm]
        rule = EmbeddingRule.from_embedders(fs)
        assert list(rule.keys()) == ts


def test_embedding_rule_assertion(image_av_chunk): # noqa

    chunk = image_av_chunk
    n_image_embed = 5
    n_av_embed = 10
    f1 = ImageEmbedder(
        RGBDImage,
        lambda img: torch.zeros(n_image_embed),
        lambda vec: torch.zeros(4, 100, 100),
        (100, 100, 4), n_image_embed)
    f2 = IdenticalEmbedder(AngleVector, n_av_embed)
    rule = EmbeddingRule.from_embedders([f1, f2])

    with pytest.raises(AssertionError):
        rule.apply_to_multi_episode_chunk(chunk)
