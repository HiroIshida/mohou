from test_encoding_rule import create_encoding_rule
from test_types import image_av_chunk_uneven  # noqa

from mohou.model.chimera import ChimeraDataset


def test_chimera_dataset(image_av_chunk_uneven):  # noqa
    chunk = image_av_chunk_uneven
    rules = create_encoding_rule(chunk, normalize=False)
    dataset = ChimeraDataset.from_chunk(chunk, rules)
    item = dataset[0]
    image_seq, vector_seq = item

    assert image_seq.ndim == 4
    assert vector_seq.ndim == 2
    assert image_seq.shape[0] == vector_seq.shape[0]  # n_seqlen equal

    n_aug = 20
    assert len(dataset) == len(chunk) * (n_aug + 1)
