import logging
from pathlib import Path
from typing import List, Optional, Type

import numpy as np

from mohou.encoder import ImageEncoder, VectorIdenticalEncoder
from mohou.encoding_rule import CovarianceBasedScaleBalancer, EncodingRule
from mohou.model import AutoEncoderBase
from mohou.propagator import PropagatorBaseT
from mohou.trainer import TrainCache
from mohou.types import (
    AngleVector,
    EpisodeBundle,
    GripperState,
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


def create_default_encoding_rule(project_path: Path) -> EncodingRule:
    bundle = EpisodeBundle.load(project_path)
    bundle_spec = bundle.spec
    av_dim = bundle_spec.type_shape_table[AngleVector][0]
    image_encoder = load_default_image_encoder(project_path)
    av_idendical_encoder = VectorIdenticalEncoder(AngleVector, av_dim)

    encoders = [image_encoder, av_idendical_encoder]

    if GripperState in bundle_spec.type_shape_table:
        gs_identital_func = VectorIdenticalEncoder(
            GripperState, bundle_spec.type_shape_table[GripperState][0]
        )
        encoders.append(gs_identital_func)

    tf_identical_func = VectorIdenticalEncoder(TerminateFlag, 1)
    encoders.append(tf_identical_func)

    p = CovarianceBasedScaleBalancer.get_json_file_path(project_path)
    if p.exists():  # use cached balacner
        logger.warning("warn: loading cached CovarianceBasedScaleBalancer")
        balancer = CovarianceBasedScaleBalancer.load(project_path)
        encoding_rule = EncodingRule.from_encoders(encoders, bundle=None, scale_balancer=balancer)
    else:
        encoding_rule = EncodingRule.from_encoders(encoders, bundle=bundle, scale_balancer=None)
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
