from typing import List, Optional, Type

import numpy as np

from mohou.encoder import ImageEncoder, VectorIdenticalEncoder
from mohou.encoding_rule import EncodingRule
from mohou.model import LSTM, AutoEncoderBase
from mohou.propagator import Propagator
from mohou.trainer import TrainCache
from mohou.types import (
    AngleVector,
    GripperState,
    MultiEpisodeChunk,
    TerminateFlag,
    get_all_concrete_leaftypes,
)


class DefaultNotFoundError(Exception):
    pass


def auto_detect_autoencoder_type(project_name: Optional[str] = None) -> Type[AutoEncoderBase]:

    # TODO(HiroIshida) dirty...
    t: Optional[Type[AutoEncoderBase]] = None

    t_cand_list = get_all_concrete_leaftypes(AutoEncoderBase)

    detect_count = 0
    for t_cand in t_cand_list:
        try:
            TrainCache.load(project_name, t_cand)
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


def load_default_image_encoder(project_name: Optional[str] = None) -> ImageEncoder:
    ae_type = auto_detect_autoencoder_type(project_name)
    try:
        tcache_autoencoder = TrainCache.load(project_name, ae_type)
    except Exception:
        raise DefaultNotFoundError("not TrainCache for autoencoder is found ")

    if tcache_autoencoder.best_model is None:
        raise DefaultNotFoundError("traincache was found but best model is not saved ")
    return tcache_autoencoder.best_model.get_encoder()


def create_default_encoding_rule(project_name: Optional[str] = None) -> EncodingRule:

    chunk = MultiEpisodeChunk.load(project_name)
    chunk_spec = chunk.spec
    av_dim = chunk_spec.type_shape_table[AngleVector][0]
    image_encoder = load_default_image_encoder(project_name)
    av_idendical_encoder = VectorIdenticalEncoder(AngleVector, av_dim)

    encoders = [image_encoder, av_idendical_encoder]

    if GripperState in chunk_spec.type_shape_table:
        gs_identital_func = VectorIdenticalEncoder(
            GripperState, chunk_spec.type_shape_table[GripperState][0]
        )
        encoders.append(gs_identital_func)

    tf_identical_func = VectorIdenticalEncoder(TerminateFlag, 1)
    encoders.append(tf_identical_func)

    encoding_rule = EncodingRule.from_encoders(encoders, chunk)
    return encoding_rule


def create_default_propagator(project_name: Optional[str] = None) -> Propagator:
    try:
        tcach_lstm = TrainCache.load(project_name, LSTM)
    except Exception:
        raise DefaultNotFoundError("not TrainCache for lstm is found ")

    encoding_rule = create_default_encoding_rule(project_name)
    assert tcach_lstm.best_model is not None
    propagator = Propagator(tcach_lstm.best_model, encoding_rule)
    return propagator


def create_default_image_context_list(project_name: Optional[str] = None) -> List[np.ndarray]:
    chunk = MultiEpisodeChunk.load(project_name)
    image_encoder = load_default_image_encoder(project_name)

    context_list = []
    for episode in chunk:
        seq = episode.get_sequence_by_type(image_encoder.elem_type)
        context = image_encoder.forward(seq[0])
        context_list.append(context)

    return context_list
