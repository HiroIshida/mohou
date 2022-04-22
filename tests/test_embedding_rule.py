import pytest

from itertools import permutations
import torch

from mohou.embedder import ImageEmbedder, IdenticalEmbedder
from mohou.embedding_rule import EmbeddingRule
from mohou.types import AngleVector, RGBImage, RGBDImage, TerminateFlag, VectorBase

from test_types import image_av_chunk # noqa


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

    rule = EmbeddingRule.from_embedders([f1, f2])
    vector_seq_list = rule.apply_to_multi_episode_chunk(chunk)
    vector_seq = vector_seq_list[0]

    assert vector_seq.shape == (10, n_image_embed + n_av_embed)

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
