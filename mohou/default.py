from typing import Type, Optional
from mohou.trainer import TrainCache
from mohou.model import AutoEncoderBase, LSTM
from mohou.types import AngleVector, GripperState, TerminateFlag, MultiEpisodeChunk, get_all_concrete_leaftypes
from mohou.embedder import IdenticalEmbedder
from mohou.embedding_rule import EmbeddingRule
from mohou.propagator import Propagator


class DefaultNotFoundError(Exception):
    pass


def auto_detect_autoencoder_type(project_name: str) -> Type[AutoEncoderBase]:
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


def create_default_embedding_rule(project_name: str) -> EmbeddingRule:

    chunk = MultiEpisodeChunk.load(project_name)
    chunk_spec = chunk.spec
    av_dim = chunk_spec.type_shape_table[AngleVector][0]
    ae_type = auto_detect_autoencoder_type(project_name)

    try:
        tcache_autoencoder = TrainCache.load(project_name, ae_type)
    except Exception:
        raise DefaultNotFoundError("not TrainCache for autoencoder is found ")

    if tcache_autoencoder.best_model is None:
        raise DefaultNotFoundError("traincache was found but best model is not saved ")
    image_embed_func = tcache_autoencoder.best_model.get_embedder()
    av_idendical_func = IdenticalEmbedder(AngleVector, av_dim)

    embedders = [image_embed_func, av_idendical_func]

    if GripperState in chunk_spec.type_shape_table:
        gs_identital_func = IdenticalEmbedder(GripperState, chunk_spec.type_shape_table[GripperState][0])
        embedders.append(gs_identital_func)

    tf_identical_func = IdenticalEmbedder(TerminateFlag, 1)
    embedders.append(tf_identical_func)

    embed_rule = EmbeddingRule.from_embedders(embedders, chunk)
    return embed_rule


def create_default_propagator(project_name: str) -> Propagator:
    try:
        tcach_lstm = TrainCache.load(project_name, LSTM)
    except Exception:
        raise DefaultNotFoundError("not TrainCache for lstm is found ")

    embed_rule = create_default_embedding_rule(project_name)
    assert tcach_lstm.best_model is not None
    propagator = Propagator(tcach_lstm.best_model, embed_rule)
    return propagator
