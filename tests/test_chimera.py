import pytest
import torch
from test_encoding_rule import create_encoding_rule
from test_types import image_av_chunk_uneven  # noqa

from mohou.model import AutoEncoderConfig, LSTMConfig
from mohou.model.autoencoder import AutoEncoder
from mohou.model.chimera import Chimera, ChimeraConfig, ChimeraDataset
from mohou.types import AngleVector, MultiEpisodeChunk, RGBImage


@pytest.fixture(scope="session")
def chimera_dataset(image_av_chunk_uneven):  # noqa
    chunk = image_av_chunk_uneven
    rule = create_encoding_rule(chunk, balance=False)
    dataset = ChimeraDataset.from_chunk(chunk, rule)
    return dataset


def test_chimera_dataset(image_av_chunk_uneven, chimera_dataset):  # noqa
    dataset = chimera_dataset
    item = dataset[0]
    image_seq, vector_seqs = item

    assert image_seq.ndim == 4
    assert vector_seqs.ndim == 3
    assert image_seq.shape[0] == vector_seqs.shape[1]  # n_seqlen equal

    n_aug = 20
    assert vector_seqs.shape[0] == (n_aug + 1)


def test_chimera_model(image_av_chunk_uneven, chimera_dataset):  # noqa
    chunk: MultiEpisodeChunk = image_av_chunk_uneven
    rule = create_encoding_rule(chunk, balance=False)

    # create config using rule info
    lstm_config = LSTMConfig(rule.dimension)
    image_shape = chunk.get_element_shape(RGBImage)
    ae_config = AutoEncoderConfig(
        RGBImage, n_bottleneck=rule[RGBImage].output_size, n_pixel=image_shape[1]
    )

    # create model
    models = []

    conf = ChimeraConfig(lstm_config, ae_config)
    models.append(Chimera(conf))  # from conf

    conf = ChimeraConfig(conf.lstm_config, AutoEncoder(ae_config))
    models.append(Chimera(conf))

    for model in models:
        n_batch = 8
        n_seqlen = 12
        n_aug = 10
        image_tensor_shape = tuple(reversed(chunk.get_element_shape(RGBImage)))
        av_dim = chunk.get_element_shape(AngleVector)[0]
        image_seqs = torch.randn((n_batch, n_seqlen, *image_tensor_shape))
        vector_seqs_seq = torch.randn((n_batch, n_aug, n_seqlen, av_dim + 1))  # 1 for terminal flag

        loss_dict = model.loss((image_seqs, vector_seqs_seq))
        keys = ["prediction", "reconstruction"]
        for key in keys:
            assert key in loss_dict
            assert loss_dict[key].item() > 0
