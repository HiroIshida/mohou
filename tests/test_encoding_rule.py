import copy
from itertools import permutations
from typing import Tuple

import numpy as np
import pytest
import torch
from test_types import image_av_bundle, rgbd_image_bundle  # noqa

from mohou.encoder import ImageEncoder, VectorIdenticalEncoder
from mohou.encoding_rule import (
    CovarianceBalancer,
    CovarianceBasedScaleBalancer,
    EncodingRule,
)
from mohou.types import (
    AngleVector,
    DepthImage,
    ElementDict,
    EpisodeBundle,
    ImageBase,
    RGBDImage,
    RGBImage,
    TerminateFlag,
    VectorBase,
)


def test_covariance_based_balancer():
    dim1 = 2
    dim2 = 3
    bias = 10
    a = np.random.randn(100000, dim1) + np.ones(2) * bias
    b = np.random.randn(100000, dim2)
    b[:, 0] *= 3
    b[:, 1] *= 2
    b[:, 2] *= 0.5
    c = np.concatenate([a, b], axis=1)
    balancer = CovarianceBasedScaleBalancer.from_feature_seqs(c, [dim1, dim2])
    inp = np.random.randn(5)
    balanced = balancer.apply(inp)
    debalanced = balancer.inverse_apply(balanced)
    np.testing.assert_almost_equal(inp, debalanced, decimal=2)

    np.testing.assert_almost_equal(balancer.scaled_stds, np.array([1.0 / 3.0, 1.0]), decimal=2)


def test_covariance_based_balancer_with_static_values():
    a = np.random.randn(1000, 3)
    a[:, 1] *= 0.0
    a[:, 2] *= 0.0
    with pytest.raises(AssertionError):
        CovarianceBasedScaleBalancer.from_feature_seqs(a, [2, 1])


class Vector1(VectorBase):
    pass


class Vector2(VectorBase):
    pass


@pytest.fixture(scope="session")
def sample_covariance_balancer():

    dim1 = 2
    dim2 = 3
    bias = 10
    a = np.random.randn(100000, dim1) + np.ones(2) * bias
    b = np.random.randn(100000, dim2)
    b[:, 0] *= 3
    b[:, 1] *= 2
    b[:, 2] *= 0.5
    c = np.concatenate([a, b], axis=1)
    balancer = CovarianceBalancer.from_feature_seqs(c, {Vector1: dim1, Vector2: dim2})
    return balancer


def test_covariance_balancer(sample_covariance_balancer):
    balancer: CovarianceBalancer = sample_covariance_balancer

    inp = np.random.randn(5)
    balanced = balancer.apply(inp)
    debalanced = balancer.inverse_apply(balanced)
    np.testing.assert_almost_equal(inp, debalanced, decimal=2)
    sp_stds = [val.scaled_primary_std for val in balancer.type_balancer_table.values()]  # type: ignore
    np.testing.assert_almost_equal(sp_stds, np.array([1.0 / 3.0, 1.0]), decimal=2)


def test_covariance_balancer_delete(sample_covariance_balancer):
    balancer: CovarianceBalancer = copy.deepcopy(sample_covariance_balancer)

    std_vec1_original = balancer.type_balancer_table[Vector1].scaled_primary_std  # type: ignore
    balancer.delete(Vector2)
    std_vec1_after = balancer.type_balancer_table[Vector1].scaled_primary_std  # type: ignore
    assert std_vec1_original < std_vec1_after

    inp = np.random.randn(2)
    balanced = balancer.apply(inp)
    debalanced = balancer.inverse_apply(balanced)
    np.testing.assert_almost_equal(inp, debalanced)


def test_covariance_balancer_marknull(sample_covariance_balancer):
    balancer: CovarianceBalancer = copy.deepcopy(sample_covariance_balancer)
    balancer.mark_null(Vector1)

    # test input output match
    inp = np.random.randn(5)
    balanced = balancer.apply(inp)
    debalanced = balancer.inverse_apply(balanced)
    np.testing.assert_almost_equal(inp, debalanced)

    # test null part will not change
    np.testing.assert_almost_equal(balanced[:2], inp[:2])

    # and vice-versa
    with pytest.raises(AssertionError):
        np.testing.assert_almost_equal(balanced[3:], inp[3:])


def test_elem_cov_match_regression_test_temporary():
    dim1 = 2
    dim2 = 3
    bias = 10
    a = np.random.randn(100000, dim1) + np.ones(2) * bias
    b = np.random.randn(100000, dim2)
    b[:, 0] *= 3
    b[:, 1] *= 2
    b[:, 2] *= 0.5
    c = np.concatenate([a, b], axis=1)
    balancer = CovarianceBalancer.from_feature_seqs(c, {Vector1: dim1, Vector2: dim2})
    balancer2 = CovarianceBasedScaleBalancer.from_feature_seqs(c, [dim1, dim2])

    for _ in range(20):
        a = np.random.randn(5)
        np.testing.assert_almost_equal(balancer.apply(a), balancer2.apply(a))


def create_encoding_rule_for_image_av_bundle(
    bundle: EpisodeBundle, balance: bool = True
) -> EncodingRule:
    dim_image_encoded = 5
    dim_av = bundle.get_element_shape(AngleVector)[0]
    image_type = [t for t in bundle.types() if issubclass(t, ImageBase)].pop()
    dims_image: Tuple[int, int, int] = bundle.get_element_shape(image_type)  # type: ignore

    def forward_impl(img_tensor: torch.Tensor):
        # whatever function as long as it's deterministic injective function
        vec = img_tensor[0, 0, 0, :dim_image_encoded].float()
        return vec

    f1 = ImageEncoder(
        image_type,
        forward_impl,
        lambda vec: torch.zeros(tuple(reversed(dims_image))),
        dims_image,
        dim_image_encoded,
    )
    f2 = VectorIdenticalEncoder(AngleVector, dim_av)
    f3 = VectorIdenticalEncoder(TerminateFlag, 1)
    optional_bundle = bundle if balance else None
    rule = EncodingRule.from_encoders([f1, f2, f3], bundle=optional_bundle)
    return rule


def test_encoding_rule_apply_to_edict(rgbd_image_bundle):  # noqa
    dim_image_encoded = 5

    def forward_impl(img_tensor: torch.Tensor):
        # whatever function as long as it's deterministic injective function
        vec = img_tensor[0, 0, 0, :dim_image_encoded].float()
        return vec

    f1 = ImageEncoder(
        RGBDImage,
        forward_impl,
        lambda vec: torch.zeros((4, 30, 30)),
        (30, 30, 4),
        5,
    )
    rule = EncodingRule.from_encoders([f1])

    rgbd = RGBDImage.dummy_from_shape((30, 30))
    rule.apply(ElementDict([rgbd]))

    rgb = RGBImage.dummy_from_shape((30, 30))
    depth = DepthImage.dummy_from_shape((30, 30))
    rule.apply(ElementDict([rgb, depth]))


def test_encoding_rule(image_av_bundle):  # noqa
    bundle = image_av_bundle
    rule = create_encoding_rule_for_image_av_bundle(bundle)
    vector_seq_list = rule.apply_to_episode_bundle(bundle)
    vector_seq = vector_seq_list[0]
    assert vector_seq.shape == (len(bundle[0]), rule.dimension)


def test_encoding_rule_type_bound_table(image_av_bundle):  # noqa
    bundle: EpisodeBundle = image_av_bundle
    rule = create_encoding_rule_for_image_av_bundle(bundle)

    last_end_idx = 0
    for elem_type, bound in rule.type_bound_table.items():
        assert bound.start == last_end_idx
        assert rule[elem_type].output_size == bound.stop - bound.start
        last_end_idx = bound.stop
    assert last_end_idx == rule.dimension


def test_encoding_rule_order(image_av_bundle):  # noqa
    class Dummy(VectorBase):
        pass

    bundle = image_av_bundle
    rule = create_encoding_rule_for_image_av_bundle(bundle)

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


def test_encoding_rule_assertion(image_av_bundle):  # noqa
    bundle = image_av_bundle
    rule = create_encoding_rule_for_image_av_bundle(bundle)
    # add wrong dimension encoder
    rule[AngleVector] = VectorIdenticalEncoder(AngleVector, 1000)

    with pytest.raises(AssertionError):
        rule.apply_to_episode_bundle(bundle)
