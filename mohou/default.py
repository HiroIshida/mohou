import logging
from functools import lru_cache
from pathlib import Path
from typing import List, Optional, Type

import numpy as np

from mohou.encoder import EncoderBase, ImageEncoder, VectorIdenticalEncoder
from mohou.encoding_rule import CovarianceBasedScaleBalancer, EncodingRule
from mohou.model import AutoEncoderBase
from mohou.model.chimera import Chimera
from mohou.propagator import Propagator, PropagatorBaseT
from mohou.trainer import TrainCache
from mohou.types import (
    AngleVector,
    AnotherGripperState,
    DepthImage,
    ElementBase,
    EpisodeBundle,
    GripperState,
    RGBDImage,
    RGBImage,
    TerminateFlag,
    get_all_concrete_leaftypes,
)

logger = logging.getLogger(__name__)


class DefaultNotFoundError(Exception):
    pass


def auto_detect_autoencoder_type(project_path: Path) -> Type[AutoEncoderBase]:
    # TODO(HiroIshida) dirty...
    t: Optional[Type[AutoEncoderBase]] = None

    t_cand_list = get_all_concrete_leaftypes(AutoEncoderBase)

    detect_count = 0
    for t_cand in t_cand_list:
        try:
            TrainCache.load(project_path, t_cand)
            t = t_cand
            detect_count += 1
        except FileNotFoundError:
            pass
        except Exception as e:
            raise e

    if detect_count == 0:
        raise DefaultNotFoundError("no autoencoder found")
    if detect_count > 1:
        raise DefaultNotFoundError("multiple autoencoder found")

    assert t is not None  # redundant but for mypy check
    return t


def load_default_image_encoder(project_path: Path) -> ImageEncoder:
    ae_type = auto_detect_autoencoder_type(project_path)
    try:
        tcache_autoencoder = TrainCache.load(project_path, ae_type)
    except Exception:
        raise DefaultNotFoundError("not TrainCache for autoencoder is found ")

    if tcache_autoencoder.best_model is None:
        raise DefaultNotFoundError("traincache was found but best model is not saved ")
    return tcache_autoencoder.best_model.get_encoder()


@lru_cache(maxsize=40)
def create_default_encoding_rule(
    project_path: Path,
    include_image_encoder: bool = True,
    use_balancer: bool = True,
) -> EncodingRule:

    bundle = EpisodeBundle.load(project_path)
    bundle_spec = bundle.spec

    encoders: List[EncoderBase] = []

    if include_image_encoder:
        image_encoder = load_default_image_encoder(project_path)
        encoders.append(image_encoder)

    if AngleVector in bundle_spec.type_shape_table:
        av_dim = bundle_spec.type_shape_table[AngleVector][0]
        av_idendical_encoder = VectorIdenticalEncoder(AngleVector, av_dim)
        encoders.append(av_idendical_encoder)

    if GripperState in bundle_spec.type_shape_table:
        gs_identital_func = VectorIdenticalEncoder(
            GripperState, bundle_spec.type_shape_table[GripperState][0]
        )
        encoders.append(gs_identital_func)

    if AnotherGripperState in bundle_spec.type_shape_table:
        ags_identital_func = VectorIdenticalEncoder(
            AnotherGripperState, bundle_spec.type_shape_table[AnotherGripperState][0]
        )
        encoders.append(ags_identital_func)

    tf_identical_func = VectorIdenticalEncoder(TerminateFlag, 1)
    encoders.append(tf_identical_func)

    p = CovarianceBasedScaleBalancer.get_json_file_path(project_path)
    if p.exists():  # use cached balacner
        if use_balancer:
            logger.warning(
                "warn: loading cached CovarianceBasedScaleBalancer. This feature is experimental."
            )
            balancer = CovarianceBasedScaleBalancer.load(project_path)
        else:
            balancer = None
        encoding_rule = EncodingRule.from_encoders(encoders, bundle=None, scale_balancer=balancer)
    else:
        if use_balancer:
            bundle_for_balancer = bundle
        else:
            bundle_for_balancer = None
        encoding_rule = EncodingRule.from_encoders(
            encoders, bundle=bundle_for_balancer, scale_balancer=None
        )

    # TODO: Move The following check to unittest? but it's diffcult becaues
    # using this function pre-require the existence of trained AE ...
    # so temporary, check using assertion

    # check if ordered propery. Keeping this order is important to satisfy the
    # backward-compatibility of this function.
    order_definition: List[Type[ElementBase]] = [
        RGBImage,
        DepthImage,
        RGBDImage,
        AngleVector,
        GripperState,
        AnotherGripperState,
        TerminateFlag,
    ]
    indices = [order_definition.index(key) for key in encoding_rule.keys()]
    is_ordered_propery = sorted(indices) == indices
    assert is_ordered_propery

    return encoding_rule


def create_default_propagator(
    project_path: Path, prop_type: Type[PropagatorBaseT]
) -> PropagatorBaseT:
    try:
        compat_lstm_type = prop_type.lstm_type()
        tcach_lstm = TrainCache.load(project_path, compat_lstm_type)
    except Exception:
        raise DefaultNotFoundError("not TrainCache for lstm is found ")

    encoding_rule = create_default_encoding_rule(project_path)
    propagator = prop_type(tcach_lstm.best_model, encoding_rule)
    return propagator


def create_default_chimera_propagator(project_path: Path):
    # TODO: move to inside of create_default_propagator

    logger.warning("warn: this feature is quite experimental. maybe deleted without notfication")

    tcache_chimera = TrainCache.load(project_path, Chimera)
    chimera_model = tcache_chimera.best_model

    rule = create_default_encoding_rule(project_path)
    rule[RGBImage] = chimera_model.ae.get_encoder()
    propagator = Propagator(chimera_model.lstm, rule)
    return propagator


def create_default_image_context_list(
    project_path: Path, bundle: Optional[EpisodeBundle] = None
) -> List[np.ndarray]:
    if bundle is None:
        bundle = EpisodeBundle.load(project_path)
    image_encoder = load_default_image_encoder(project_path)

    context_list = []
    for episode in bundle:
        seq = episode.get_sequence_by_type(image_encoder.elem_type)
        context = image_encoder.forward(seq[0])
        context_list.append(context)

    return context_list
