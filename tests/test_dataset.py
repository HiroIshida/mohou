import torch

from mohou.embedder import ImageEmbedder, IdenticalEmbedder
from mohou.embedding_rule import EmbeddingRule
from mohou.types import AngleVector, RGBImage
from mohou.dataset import AutoRegressiveDataset, AutoRegressiveDatasetConfig

from test_types import image_av_chunk # noqa


def test_embedding_rule_assertion(image_av_chunk): # noqa
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

    n_aug = 7
    config = AutoRegressiveDatasetConfig(n_aug)
    dataset = AutoRegressiveDataset.from_chunk(chunk, rule, config)
    assert len(dataset.state_seq_list) == len(chunk.data_list) * (n_aug + 1)
