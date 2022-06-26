import pytest
import torch
from test_encoding_rule import create_encoding_rule
from test_types import image_av_chunk_uneven  # noqa

from mohou.model import AutoEncoderConfig, LSTMConfig
from mohou.model.chimera import Chimera, ChimeraConfig, ChimeraDataset
from mohou.types import AngleVector, MultiEpisodeChunk, RGBImage


@pytest.fixture(scope="session")
def chimera_dataset(image_av_chunk_uneven):  # noqa
    chunk = image_av_chunk_uneven
    rule = create_encoding_rule(chunk, normalize=False)
    dataset = ChimeraDataset.from_chunk(chunk, rule)
    return dataset


def test_chimera_dataset(image_av_chunk_uneven, chimera_dataset):  # noqa
    chunk = image_av_chunk_uneven
    dataset = chimera_dataset
    item = dataset[0]
    image_seq, vector_seq = item

    assert image_seq.ndim == 4
    assert vector_seq.ndim == 2
    assert image_seq.shape[0] == vector_seq.shape[0]  # n_seqlen equal

    n_aug = 20
    assert len(dataset) == len(chunk) * (n_aug + 1)


def test_chimera(image_av_chunk_uneven, chimera_dataset):  # noqa
    chunk: MultiEpisodeChunk = image_av_chunk_uneven
    rule = create_encoding_rule(chunk, normalize=False)

    # create config using rule info
    lstm_config = LSTMConfig(rule.dimension)
    image_shape = chunk.get_element_shape(RGBImage)
    ae_config = AutoEncoderConfig(
        RGBImage, n_bottleneck=rule[RGBImage].output_size, n_pixel=image_shape[1]
    )
    conf = ChimeraConfig(lstm_config, ae_config)

    # create model
    model = Chimera(conf)

    n_batch = 8
    n_seqlen = 12
    image_tensor_shape = tuple(reversed(chunk.get_element_shape(RGBImage)))
    av_dim = chunk.get_element_shape(AngleVector)[0]
    image_seqs = torch.randn((n_batch, n_seqlen, *image_tensor_shape))
    vector_seqs = torch.randn((n_batch, n_seqlen, av_dim + 1))  # 1 for terminal flag

    loss_dict = model.loss((image_seqs, vector_seqs))
    keys = ["prediction", "reconstruction"]
    for key in keys:
        assert key in loss_dict
        assert loss_dict[key].item() > 0
