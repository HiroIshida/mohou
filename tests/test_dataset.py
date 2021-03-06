import numpy as np
from test_encoding_rule import create_encoding_rule  # noqa
from test_types import image_av_bundle_uneven  # noqa
from torch.utils.data import DataLoader

from mohou.dataset import (
    AutoEncoderDataset,
    AutoEncoderDatasetConfig,
    AutoRegressiveDataset,
    AutoRegressiveDatasetConfig,
    MarkovControlSystemDataset,
    SequenceDatasetConfig,
)
from mohou.dataset.sequence_dataset import PaddingSequenceAligner, SequenceDataAugmentor
from mohou.encoding_rule import EncodingRule
from mohou.types import AngleVector, EpisodeBundle, RGBImage
from mohou.utils import assert_seq_list_list_compatible


def test_autoencoder_dataset(image_av_bundle_uneven):  # noqa

    n_image_original = 0
    for episode_data in image_av_bundle_uneven:
        n_image_original += len(episode_data.get_sequence_by_type(RGBImage))

    config = AutoEncoderDatasetConfig(batch_augment_factor=4)
    dataset = AutoEncoderDataset.from_bundle(image_av_bundle_uneven, RGBImage, config)

    train_loader = DataLoader(dataset=dataset, batch_size=3, shuffle=True)
    n_sample_total = 0
    for samples in train_loader:
        n_sample_total += samples.shape[0]
    assert n_sample_total == n_image_original * (config.batch_augment_factor + 1)


def test_sequence_data_augmentor():
    cov_scale = 0.9
    config = SequenceDatasetConfig(n_aug=1, cov_scale=cov_scale)

    cov_grount_truth = np.diag([2**2, 3**2])

    def creat_random_walk(n_seqlen: int) -> np.ndarray:
        x = np.random.randn(2)
        x_list = [x]
        for i in range(n_seqlen):
            mean = np.zeros(2)
            noise = np.random.multivariate_normal(mean, cov_grount_truth, 1)
            x_list.append(x_list[-1] + noise.flatten())
        return np.array(x_list)

    random_walks = []
    for _ in range(50):
        n_seqlen = 100 + np.random.randint(10)  # real data length is different from seq to seq
        random_walks.append(creat_random_walk(n_seqlen))

    augmentor = SequenceDataAugmentor.from_seqs(random_walks, config)

    # check if cov computed from seqs matches with the original
    diff = np.abs(augmentor.covmat - cov_grount_truth)
    assert np.max(diff) < 1.0

    auged_seqs = augmentor.apply(np.zeros((1000, 2)))
    auged_seqs.pop(0)

    cov_scaled_ground_trugh = cov_grount_truth * cov_scale**2
    for seq in auged_seqs:
        covmat = np.cov(seq.T)
        diff = np.abs(covmat - cov_scaled_ground_trugh)
        assert np.max(diff) < 1.0


def test_padding_sequnece_alginer():
    # simple test
    aligner = PaddingSequenceAligner(10)
    assert aligner.apply([1, 2, 3, 4]) == [1, 2, 3, 4, 4, 4, 4, 4, 4, 4]

    # automatic test
    def test_inner(seq):
        is_numpy = isinstance(seq[0], np.ndarray)

        n_seqlen = len(seq)
        n_seqlen_target = 10
        aligner = PaddingSequenceAligner(n_seqlen_target)
        seq_padded = aligner.apply(seq)

        for i in range(n_seqlen_target):
            if i < n_seqlen:
                if is_numpy:
                    np.testing.assert_almost_equal(seq_padded[i], seq[i])
                else:
                    assert seq_padded[i] == seq[i]
            else:
                if is_numpy:
                    np.testing.assert_almost_equal(seq_padded[i], seq[n_seqlen - 1])
                else:
                    assert seq_padded[i] == seq[n_seqlen - 1]

    test_inner([1, 2, 3, 4])
    test_inner([np.random.randn() for _ in range(4)])
    test_inner([np.random.randn(3) for _ in range(4)])
    test_inner([np.random.randn(3, 3) for _ in range(4)])
    test_inner([np.random.randn(3, 3, 3) for _ in range(4)])


def test_padding_sequnece_alginer_creation():
    n_seq = 5
    n_seqlen_max = -1
    seqs = []
    for i in range(n_seq):
        n_seqlen = 10 + np.random.randint(10)
        n_seqlen_max = max(n_seqlen, n_seqlen_max)
        seqs.append(np.array([np.random.randn() for _ in range(n_seqlen)]))

    n_after = 2
    aligner = PaddingSequenceAligner.from_seqs(seqs, n_after)
    aligner.n_seqlen_target == n_seqlen_max + n_after


def test_auto_regressive_dataset(image_av_bundle_uneven):  # noqa
    bundle: EpisodeBundle = image_av_bundle_uneven
    rule = create_encoding_rule(bundle, balance=False)

    n_aug = 7
    config = AutoRegressiveDatasetConfig(n_aug=n_aug, cov_scale=0.1)
    dataset = AutoRegressiveDataset.from_bundle(bundle, rule, config)
    assert len(dataset.state_seq_list) == len(bundle.get_touch_bundle()) * (n_aug + 1)
    assert_seq_list_list_compatible([dataset.state_seq_list, dataset.weight_seq_list])

    for state_seq in dataset.state_seq_list:
        n_length, n_dim = state_seq.shape
        assert n_length == 13 + config.n_dummy_after_termination
        assert n_dim == rule.dimension


def test_markov_control_system_dataset(image_av_bundle_uneven):  # noqa
    bundle: EpisodeBundle = image_av_bundle_uneven
    default_rule = create_encoding_rule(bundle, balance=False)
    f1 = default_rule[RGBImage]
    f2 = default_rule[AngleVector]
    control_encode_rule = EncodingRule.from_encoders([f2])
    observation_encode_rule = EncodingRule.from_encoders([f1, f2])

    n_aug = 20
    config = SequenceDatasetConfig(n_aug=n_aug, cov_scale=0.0)  # 0.0 to remove noise
    for diff_as_control in [True, False]:
        dataset = MarkovControlSystemDataset.from_bundle(
            bundle,
            control_encode_rule,
            observation_encode_rule,
            diff_as_control=diff_as_control,
            config=config,
        )

        controls_seq = control_encode_rule.apply_to_episode_bundle(bundle)
        observations_seq = observation_encode_rule.apply_to_episode_bundle(bundle)

        # test __len__
        n_len_ground_truth = sum([len(seq) - 1 for seq in controls_seq]) * (n_aug + 1)
        assert len(dataset) == n_len_ground_truth

        # test the first content (because cov = 0.0 ..)
        inp_ctrl, inp_obs, out_obs = dataset[0]

        if diff_as_control:
            np.testing.assert_almost_equal(inp_ctrl, controls_seq[0][1] - controls_seq[0][0])
        else:
            np.testing.assert_almost_equal(inp_ctrl, controls_seq[0][0])
        np.testing.assert_almost_equal(inp_obs, observations_seq[0][0])
        np.testing.assert_almost_equal(out_obs, observations_seq[0][1])

        # test the last content (because cov = 0.0 ...)
        inp_ctrl, inp_obs, out_obs = dataset[-1]
        if diff_as_control:
            np.testing.assert_almost_equal(inp_ctrl, controls_seq[-1][1] - controls_seq[-1][0])
        else:
            np.testing.assert_almost_equal(inp_ctrl, controls_seq[-1][0])
        np.testing.assert_almost_equal(inp_obs, observations_seq[-1][-1])
        np.testing.assert_almost_equal(out_obs, observations_seq[-1][-2])
