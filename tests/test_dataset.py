import numpy as np
import torch
from test_types import image_av_chunk_uneven  # noqa
from torch.utils.data import DataLoader

from mohou.dataset import (
    AutoEncoderDataset,
    AutoEncoderDatasetConfig,
    AutoRegressiveDataset,
    AutoRegressiveDatasetConfig,
    MarkovControlSystemDataset,
)
from mohou.encoder import ImageEncoder, VectorIdenticalEncoder
from mohou.encoding_rule import EncodingRule
from mohou.types import AngleVector, MultiEpisodeChunk, RGBImage, TerminateFlag
from mohou.utils import assert_two_sequences_same_length


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
    n_image_encoded = 5
    n_av_encoded = 10
    f1 = ImageEncoder(
        RGBImage,
        lambda img: torch.zeros(n_image_encoded),
        lambda vec: torch.zeros(3, 100, 100),
        (100, 100, 3),
        n_image_encoded,
    )
    f2 = VectorIdenticalEncoder(AngleVector, n_av_encoded)
    f3 = VectorIdenticalEncoder(TerminateFlag, 1)

    rule = EncodingRule.from_encoders([f1, f2, f3])

    n_aug = 7
    config = AutoRegressiveDatasetConfig(n_aug)
    dataset = AutoRegressiveDataset.from_chunk(chunk, rule, config)
    assert len(dataset.state_seq_list) == len(chunk.data_list) * (n_aug + 1)
    assert_two_sequences_same_length(dataset.state_seq_list, dataset.weight_seq_list)

    for state_seq in dataset.state_seq_list:
        n_length, n_dim = state_seq.shape
        assert n_length == 13 + config.n_dummy_after_termination
        assert n_dim == rule.dimension


def test_markov_control_system_dataset(image_av_chunk_uneven):  # noqa
    chunk: MultiEpisodeChunk = image_av_chunk_uneven
    n_image_encoded = 5
    n_av_encoded = 10
    f1 = ImageEncoder(
        RGBImage,
        lambda img: torch.zeros(n_image_encoded),
        lambda vec: torch.zeros(3, 100, 100),
        (100, 100, 3),
        n_image_encoded,
    )
    f2 = VectorIdenticalEncoder(AngleVector, n_av_encoded)
    control_encode_rule = EncodingRule.from_encoders([f2])
    observation_encode_rule = EncodingRule.from_encoders([f1, f2])

    for diff_as_control in [True, False]:
        dataset = MarkovControlSystemDataset.from_chunk(
            chunk, control_encode_rule, observation_encode_rule, diff_as_control=diff_as_control
        )

        controls_seq = control_encode_rule.apply_to_multi_episode_chunk(chunk)
        observations_seq = observation_encode_rule.apply_to_multi_episode_chunk(chunk)

        # test __len__
        n_len_ground_truth = sum([len(seq) - 1 for seq in controls_seq])
        assert len(dataset) == n_len_ground_truth

        # test the first content
        inp_ctrl, inp_obs, out_obs = dataset[0]

        if diff_as_control:
            np.testing.assert_almost_equal(inp_ctrl, controls_seq[0][1] - controls_seq[0][0])
        else:
            np.testing.assert_almost_equal(inp_ctrl, controls_seq[0][0])
        np.testing.assert_almost_equal(inp_obs, observations_seq[0][0])
        np.testing.assert_almost_equal(out_obs, observations_seq[0][1])

        # test the last content
        inp_ctrl, inp_obs, out_obs = dataset[-1]
        if diff_as_control:
            np.testing.assert_almost_equal(inp_ctrl, controls_seq[-1][1] - controls_seq[-1][0])
        else:
            np.testing.assert_almost_equal(inp_ctrl, controls_seq[-1][0])
        np.testing.assert_almost_equal(inp_obs, observations_seq[-1][-1])
        np.testing.assert_almost_equal(out_obs, observations_seq[-1][-2])
