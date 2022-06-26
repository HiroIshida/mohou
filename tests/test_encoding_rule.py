from itertools import permutations
from typing import Tuple

import numpy as np
import pytest
import torch
from test_types import image_av_chunk  # noqa

from mohou.encoder import ImageEncoder, VectorIdenticalEncoder
from mohou.encoding_rule import ElemCovMatchPostProcessor, EncodingRule
from mohou.types import (
    AngleVector,
    ImageBase,
    MultiEpisodeChunk,
    TerminateFlag,
    VectorBase,
)


def test_ElemCovMatchPostProcessor():
    dim1 = 2
    dim2 = 3
    bias = 10
    a = np.random.randn(100000, dim1) + np.ones(2) * bias
    b = np.random.randn(100000, dim2)
    b[:, 0] *= 3
    b[:, 1] *= 2
    b[:, 2] *= 0.5
    c = np.concatenate([a, b], axis=1)
    normalizer = ElemCovMatchPostProcessor.from_feature_seqs(c, [dim1, dim2])
    inp = np.random.randn(5)
    normalized = normalizer.apply(inp)
    denormalized = normalizer.inverse_apply(normalized)
    np.testing.assert_almost_equal(inp, denormalized, decimal=2)

    cstds = normalizer.characteristic_stds
    np.testing.assert_almost_equal(cstds, np.array([1.0, 3.0]), decimal=1)
    scaled_cstds = normalizer.scaled_characteristic_stds
    np.testing.assert_almost_equal(scaled_cstds, np.array([1.0 / 3.0, 1.0]), decimal=2)


def create_encoding_rule(chunk: MultiEpisodeChunk, normalize: bool = True) -> EncodingRule:
    dim_image_encoded = 5
    dim_av = chunk.get_element_shape(AngleVector)[0]
    image_type = [t for t in chunk.types() if issubclass(t, ImageBase)].pop()
    dims_image: Tuple[int, int, int] = chunk.get_element_shape(image_type)  # type: ignore

    f1 = ImageEncoder(
        image_type,
        lambda img: torch.zeros(dim_image_encoded),
        lambda vec: torch.zeros(tuple(reversed(dims_image))),
        dims_image,
        dim_image_encoded,
    )
    f2 = VectorIdenticalEncoder(AngleVector, dim_av)
    f3 = VectorIdenticalEncoder(TerminateFlag, 1)
    optional_chunk = chunk if normalize else None
    rule = EncodingRule.from_encoders([f1, f2, f3], chunk=optional_chunk)
    return rule


def test_encoding_rule(image_av_chunk):  # noqa
    chunk = image_av_chunk
    rule = create_encoding_rule(chunk)
    vector_seq_list = rule.apply_to_multi_episode_chunk(chunk)
    vector_seq = vector_seq_list[0]
    assert vector_seq.shape == (len(chunk[0]), rule.dimension)


def test_encoding_rule_order(image_av_chunk):  # noqa
    class Dummy(VectorBase):
        pass

    chunk = image_av_chunk
    rule = create_encoding_rule(chunk)

    # Check dict insertion oreder is preserved
    # NOTE: from 3.7, order is preserved as lang. spec.
    # NOTE: from 3.6, order is preserved in a cpython implementation
    f4 = VectorIdenticalEncoder(Dummy, 2)
    pairs = [(t, rule[t]) for t in rule.keys()]
    pairs.append((Dummy, f4))
    for pairs_perm in permutations(pairs, 4):
        types = [p[0] for p in pairs_perm]
        encoders = [p[1] for p in pairs_perm]
        rule = EncodingRule.from_encoders(encoders)
        assert rule.encode_order == types


def test_encoding_rule_assertion(image_av_chunk):  # noqa
    chunk = image_av_chunk
    rule = create_encoding_rule(chunk)
    # add wrong dimension encoder
    rule[AngleVector] = VectorIdenticalEncoder(AngleVector, 1000)

    with pytest.raises(AssertionError):
        rule.apply_to_multi_episode_chunk(chunk)
