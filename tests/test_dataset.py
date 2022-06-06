import torch
from torch.utils.data import DataLoader

from mohou.embedder import ImageEncoder, VectorIdenticalEncoder
from mohou.embedding_rule import EncodeRule
from mohou.types import AngleVector, RGBImage, TerminateFlag
from mohou.dataset import AutoEncoderDataset, AutoEncoderDatasetConfig
from mohou.dataset import AutoRegressiveDataset, AutoRegressiveDatasetConfig
from mohou.utils import assert_two_sequences_same_length

from test_types import image_av_chunk_uneven  # noqa


def test_autoencoder_dataset(image_av_chunk_uneven):  # noqa

    n_image_original = 0
    for episode_data in image_av_chunk_uneven:
        n_image_original += len(episode_data.get_sequence_by_type(RGBImage))

    config = AutoEncoderDatasetConfig(batch_augment_factor=4)
    dataset = AutoEncoderDataset.from_chunk(image_av_chunk_uneven, RGBImage, config)

    train_loader = DataLoader(dataset=dataset, batch_size=3, shuffle=True)
    n_sample_total = 0
    for samples in train_loader:
        n_sample_total += samples.shape[0]
    assert n_sample_total == n_image_original * (config.batch_augment_factor + 1)


def test_auto_regressive_dataset(image_av_chunk_uneven):  # noqa
    chunk = image_av_chunk_uneven
    n_image_embed = 5
    n_av_embed = 10
    f1 = ImageEncoder(
        RGBImage,
        lambda img: torch.zeros(n_image_embed),
        lambda vec: torch.zeros(3, 100, 100),
        (100, 100, 3),
        n_image_embed,
    )
    f2 = VectorIdenticalEncoder(AngleVector, n_av_embed)
    f3 = VectorIdenticalEncoder(TerminateFlag, 1)

    rule = EncodeRule.from_encoders([f1, f2, f3])

    n_aug = 7
    config = AutoRegressiveDatasetConfig(n_aug)
    dataset = AutoRegressiveDataset.from_chunk(chunk, rule, config)
    assert len(dataset.state_seq_list) == len(chunk.data_list) * (n_aug + 1)
    assert_two_sequences_same_length(dataset.state_seq_list, dataset.weight_seq_list)

    for state_seq in dataset.state_seq_list:
        n_length, n_dim = state_seq.shape
        assert n_length == 13 + config.n_dummy_after_termination
        assert n_dim == rule.dimension
