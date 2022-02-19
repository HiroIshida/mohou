import torch

from mohou.embedding_functor import ImageEmbeddingFunctor
from mohou.embedding_rule import RGBAngelVectorEmbeddingRule, IdenticalEmbeddingFunctor

from test_types import image_av_chunk # noqa


def test_embedding_rule(image_av_chunk): # noqa
    chunk = image_av_chunk
    f1 = ImageEmbeddingFunctor(lambda img: torch.zeros(10), (100, 100, 3), 10)
    f2 = IdenticalEmbeddingFunctor(10)

    rule = RGBAngelVectorEmbeddingRule(f1, f2)
    rule.apply_to_multi_episode_chunk(chunk)
