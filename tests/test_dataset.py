import torch

from mohou.embedder import ImageEmbedder, IdenticalEmbedder
from mohou.embedding_rule import EmbeddingRule
from mohou.types import AngleVector, RGBImage, TerminateFlag
from mohou.dataset import AutoRegressiveDataset, AutoRegressiveDatasetConfig

from test_types import image_av_chunk_uneven # noqa


def test_auto_regressive_dataset(image_av_chunk_uneven): # noqa
    chunk = image_av_chunk_uneven
    n_image_embed = 5
    n_av_embed = 10
    f1 = ImageEmbedder(
        RGBImage,
        lambda img: torch.zeros(n_image_embed),
        lambda vec: torch.zeros(3, 100, 100),
        (100, 100, 3), n_image_embed)
    f2 = IdenticalEmbedder(AngleVector, n_av_embed)
    f3 = IdenticalEmbedder(TerminateFlag, 1)

    rule = EmbeddingRule.from_embedders([f1, f2, f3])

    n_aug = 7
    config = AutoRegressiveDatasetConfig(n_aug)
    dataset = AutoRegressiveDataset.from_chunk(chunk, rule, config)
    assert len(dataset.state_seq_list) == len(chunk.data_list) * (n_aug + 1)

    for state_seq in dataset.state_seq_list:
        n_length, n_dim = state_seq.shape
        assert n_length == 13
        assert n_dim == rule.dimension
