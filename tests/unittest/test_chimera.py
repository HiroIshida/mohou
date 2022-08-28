from typing import List

import pytest
import torch
from test_encoding_rule import create_encoding_rule_for_image_av_bundle
from test_types import image_av_bundle_uneven  # noqa

from mohou.dataset import ChimeraDataset
from mohou.default import VectorIdenticalEncoder
from mohou.encoding_rule import EncodingRule
from mohou.model import AutoEncoderConfig, LSTMConfig
from mohou.model.autoencoder import AutoEncoder
from mohou.model.chimera import Chimera, ChimeraConfig
from mohou.types import AngleVector, EpisodeBundle, RGBImage, TerminateFlag


@pytest.fixture(scope="session")
def chimera_dataset(image_av_bundle_uneven):  # noqa
    bundle = image_av_bundle_uneven
    dim_av = bundle.get_element_shape(AngleVector)[0]
    f1 = VectorIdenticalEncoder(AngleVector, dim_av)
    f2 = VectorIdenticalEncoder(TerminateFlag, 1)
    rule = EncodingRule.from_encoders([f1, f2])
    dataset = ChimeraDataset.from_bundle(bundle, rule)
    return dataset


def test_chimera_dataset(image_av_bundle_uneven, chimera_dataset):  # noqa
    bundle = image_av_bundle_uneven
    dataset = chimera_dataset
    item = dataset[0]
    image_seq, vector_seq = item

    assert image_seq.ndim == 4
    assert vector_seq.ndim == 2
    assert image_seq.shape[0] == vector_seq.shape[0]  # n_seqlen equal

    n_aug = 20
    assert len(dataset) == len(bundle) * (n_aug + 1)


def test_chimera_model(image_av_bundle_uneven, chimera_dataset):  # noqa
    bundle: EpisodeBundle = image_av_bundle_uneven
    rule = create_encoding_rule_for_image_av_bundle(bundle, balance=False)

    # create config using rule info
    lstm_config = LSTMConfig(rule.dimension)
    image_shape = bundle.get_element_shape(RGBImage)
    ae_config = AutoEncoderConfig(
        RGBImage, n_bottleneck=rule[RGBImage].output_size, n_pixel=image_shape[1]
    )

    # create model
    models: List[Chimera] = []

    conf = ChimeraConfig(lstm_config, ae_config)
    models.append(Chimera(conf))  # from conf

    conf = ChimeraConfig(conf.lstm_config, AutoEncoder(ae_config))
    models.append(Chimera(conf))

    for model in models:
        model.put_on_device()
        n_batch = 8
        n_seqlen = 12
        image_tensor_shape = tuple(reversed(bundle.get_element_shape(RGBImage)))
        av_dim = bundle.get_element_shape(AngleVector)[0]
        image_seqs = torch.randn((n_batch, n_seqlen, *image_tensor_shape)).to(model.device)
        vector_seqs = torch.randn((n_batch, n_seqlen, av_dim + 1)).to(
            model.device
        )  # 1 for terminal flag

        loss_dict = model.loss((image_seqs, vector_seqs))
        keys = ["prediction", "reconstruction"]
        for key in keys:
            assert key in loss_dict
            assert loss_dict[key].item() > 0
