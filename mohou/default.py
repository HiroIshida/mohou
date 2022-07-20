from pathlib import Path
from typing import List, Optional, Type

import numpy as np

from mohou.encoder import ImageEncoder, VectorIdenticalEncoder
from mohou.encoding_rule import EncodingRule
from mohou.model import LSTM, AutoEncoderBase, Chimera
from mohou.propagator import Propagator
from mohou.trainer import TrainCache
from mohou.types import (
    AngleVector,
    EpisodeBundle,
    GripperState,
    ImageBase,
    TerminateFlag,
    get_all_concrete_leaftypes,
)


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
        except Exception:
            pass

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

    encoding_rule = EncodingRule.from_encoders(encoders, bundle)
    return encoding_rule


def create_chimera_encoding_rule(project_path: Path) -> EncodingRule:
    # experimental
    encoding_rule = create_default_encoding_rule(project_path)
    image_type = [k for k in encoding_rule.keys() if issubclass(k, ImageBase)].pop()
    chimera = TrainCache.load(project_path, Chimera).best_model
    assert chimera is not None
    image_encoder_new = chimera.get_encoder()
    assert encoding_rule[image_type].input_shape == image_encoder_new.input_shape
    assert encoding_rule[image_type].output_size == image_encoder_new.output_size

    # TODO: probably we should create a method grouning the following two
    encoding_rule[image_type] = image_encoder_new
    encoding_rule.covariance_balancer.mark_null(image_type)

    return encoding_rule


def create_default_propagator(project_path: Path) -> Propagator:

    try:
        tcach_lstm = TrainCache.load(project_path, LSTM)
    except Exception:
        raise DefaultNotFoundError("not TrainCache for lstm is found ")

    encoding_rule = create_default_encoding_rule(project_path)
    assert tcach_lstm.best_model is not None
    propagator = Propagator(tcach_lstm.best_model, encoding_rule)
    return propagator


def create_chimera_propagator(project_path: Path) -> Propagator:
    encoding_rule = create_chimera_encoding_rule(project_path)

    chimera = TrainCache.load(project_path, Chimera).best_model
    assert chimera is not None

    propagator = Propagator(chimera.lstm, encoding_rule)
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
