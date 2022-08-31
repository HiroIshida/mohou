import tempfile
from itertools import permutations
from pathlib import Path
from typing import Tuple

import numpy as np
import pytest
import torch
from test_types import image_av_bundle, rgbd_image_bundle  # noqa

from mohou.encoder import ImageEncoder, VectorIdenticalEncoder
from mohou.encoding_rule import (
    CompositeEncodingRule,
    CovarianceBasedScaleBalancer,
    EncodingRule,
    IdenticalScaleBalancer,
)
from mohou.model import AutoEncoder, AutoEncoderConfig
from mohou.types import (
    AngleVector,
    DepthImage,
    ElementDict,
    EpisodeBundle,
    RGBDImage,
    RGBImage,
    TerminateFlag,
    VectorBase,
)


def test_identical_balancer():
    balancer = IdenticalScaleBalancer()

    for _ in range(10):
        vec = np.random.randn(5)
        vec_applied = balancer.apply(vec)
        np.testing.assert_equal(vec, vec_applied)

        vec_again = balancer.inverse_apply(vec_applied)
        np.testing.assert_equal(vec_again, vec)

    with tempfile.TemporaryDirectory() as dname:
        project_path = Path(dname)
        balancer.dump(project_path)
        loaded = IdenticalScaleBalancer.load(project_path)
    assert balancer == loaded


def test_covariance_based_balancer():
    dim1 = 2
    dim2 = 3
    bias = 10
    a = np.random.randn(100000, dim1) + np.ones(dim1) * bias
    b = np.random.randn(100000, dim2)
    b[:, 0] *= 3
    b[:, 1] *= 2
    b[:, 2] *= 0.5
    c = np.concatenate([a, b], axis=1)
    balancer = CovarianceBasedScaleBalancer.from_feature_seqs(c, [dim1, dim2])
    inp = np.random.randn(5)
    balanced = balancer.apply(inp)
    debalanced = balancer.inverse_apply(balanced)
    np.testing.assert_almost_equal(inp, debalanced, decimal=8)
    np.testing.assert_almost_equal(balancer.scaled_stds, np.array([1.0 / 3.0, 1.0]), decimal=2)

    # apply to 2d arr
    n_batch = 20
    inp_arr = np.random.randn(n_batch, 5)
    balanced_arr = balancer.apply(inp_arr)
    debalanced_arr = balancer.inverse_apply(balanced_arr)
    np.testing.assert_almost_equal(inp_arr, debalanced_arr, decimal=8)
    for i in range(n_batch):
        np.testing.assert_almost_equal(balanced_arr[i], balancer.apply(inp_arr[i]))

    # apply to 2d tensor
    inp_torch = torch.from_numpy(inp_arr)
    balanced_torch = balancer.apply(inp_torch)
    debalanced_torch = balancer.inverse_apply(balanced_torch)
    np.testing.assert_almost_equal(balanced_arr, balanced_torch.detach().numpy(), decimal=4)
    torch.allclose(inp_torch, debalanced_torch)


def test_covariance_based_balancer_dump_and_load():
    dim1 = 2
    dim2 = 3
    a = np.random.randn(100000, dim1) + np.ones(dim1)
    b = np.random.randn(100000, dim2) * 10
    c = np.concatenate([a, b], axis=1)
    balancer = CovarianceBasedScaleBalancer.from_feature_seqs(c, [dim1, dim2])
    with tempfile.TemporaryDirectory() as dname:
        project_path = Path(dname)
        balancer.dump(project_path)
        loaded = CovarianceBasedScaleBalancer.load(project_path)
    assert balancer == loaded


def test_covariance_based_balancer_with_static_values():
    a = np.random.randn(1000, 3)
    a[:, 1] *= 0.0
    a[:, 2] *= 0.0
    with pytest.raises(AssertionError):
        CovarianceBasedScaleBalancer.from_feature_seqs(a, [2, 1])


def create_encoding_rule_for_image_av_bundle(
    bundle: EpisodeBundle, balance: bool = True
) -> EncodingRule:
    dim_av = bundle.get_element_shape(AngleVector)[0]
    dims_image: Tuple[int, int, int] = bundle.get_element_shape(RGBImage)  # type: ignore

    model = AutoEncoder[RGBImage](AutoEncoderConfig(RGBImage, 10, dims_image[0]))
    f1 = ImageEncoder.from_auto_encoder(model)
    f2 = VectorIdenticalEncoder.create(AngleVector, dim_av)
    f3 = VectorIdenticalEncoder.create(TerminateFlag, 1)
    optional_bundle = bundle if balance else None
    rule = EncodingRule.from_encoders([f1, f2, f3], bundle=optional_bundle)
    return rule


def test_encoding_rule_apply_to_edict(rgbd_image_bundle):  # noqa
    pass

    model = AutoEncoder[RGBDImage](AutoEncoderConfig(RGBDImage, 10, 28))
    f1 = ImageEncoder.from_auto_encoder(model)
    rule = EncodingRule.from_encoders([f1])

    rgbd = RGBDImage.dummy_from_shape((28, 28))
    rule.apply(ElementDict([rgbd]))

    rgb = RGBImage.dummy_from_shape((28, 28))
    depth = DepthImage.dummy_from_shape((28, 28))
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
    f4 = VectorIdenticalEncoder.create(Dummy, 2)
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
    rule[AngleVector] = VectorIdenticalEncoder.create(AngleVector, 1000)

    with pytest.raises(AssertionError):
        rule.apply_to_episode_bundle(bundle)


def test_composite_encoding_rule(image_av_bundle: EpisodeBundle):  # noqa
    class Dummy1(VectorBase):
        pass

    f1 = VectorIdenticalEncoder.create(Dummy1, 7)
    dummy_rule1 = EncodingRule.from_encoders([f1])

    image_av_rule = create_encoding_rule_for_image_av_bundle(image_av_bundle, balance=True)

    class Dummy2(VectorBase):
        pass

    f2 = VectorIdenticalEncoder.create(Dummy2, 4)
    dummy_rule2 = EncodingRule.from_encoders([f2])

    composite_rule = CompositeEncodingRule([dummy_rule1, image_av_rule, dummy_rule2])

    dummy1 = Dummy1(np.random.randn(7))
    rgb = image_av_bundle[0][0][RGBImage]
    av = image_av_bundle[0][0][AngleVector]
    flag = TerminateFlag.from_bool(True)
    dummy2 = Dummy2(np.random.randn(4))
    edict = ElementDict([dummy1, rgb, av, flag, dummy2])

    vec = composite_rule.apply(edict)
    edict_again = composite_rule.inverse_apply(vec)

    # reconstructed vectors must match. (but rgb may not match beacuase autoencoder is lossy compressor)
    assert edict_again[Dummy1] == edict[Dummy1]
    assert edict_again[AngleVector] == edict[AngleVector]
    assert edict_again[Dummy2] == edict[Dummy2]
    assert edict_again[TerminateFlag] == edict[TerminateFlag]

    # check key is in order
    keys = list(composite_rule.keys())
    keys_ground_truth = [Dummy1] + list(image_av_rule.keys()) + [Dummy2]  # type: ignore
    assert keys == keys_ground_truth
