import pytest

import torch

from mohou.embedder import ImageEmbedder, IdenticalEmbedder
from mohou.embedding_rule import create_embedding_rule
from mohou.types import AngleVector, RGBImage, RGBDImage

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

    rule = create_embedding_rule(f1, f2)
    vector_seq_list = rule.apply_to_multi_episode_chunk(chunk)
    vector_seq = vector_seq_list[0]

    assert vector_seq.shape == (10, n_image_embed + n_av_embed)


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
    rule = create_embedding_rule(f1, f2)

    with pytest.raises(AssertionError):
        rule.apply_to_multi_episode_chunk(chunk)
