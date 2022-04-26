from typing import Type
from mohou.trainer import TrainCache
from mohou.model import AutoEncoderBase, AutoEncoder, LSTM
from mohou.types import AngleVector, TerminateFlag
from mohou.embedder import IdenticalEmbedder
from mohou.embedding_rule import EmbeddingRule
from mohou.propagator import Propagator


def create_default_embedding_rule(
        project_name: str,
        n_angle_vector: int,
        ae_type: Type[AutoEncoderBase] = AutoEncoder) -> EmbeddingRule:

    tcache_autoencoder = TrainCache.load(project_name, ae_type)
    assert tcache_autoencoder.best_model is not None
    image_embed_func = tcache_autoencoder.best_model.get_embedder()
    av_idendical_func = IdenticalEmbedder(AngleVector, n_angle_vector)
    ef_identical_func = IdenticalEmbedder(TerminateFlag, 1)
    embed_rule = EmbeddingRule.from_embedders(
        [image_embed_func, av_idendical_func, ef_identical_func])
    return embed_rule


def create_default_propagator(
        project_name: str,
        n_angle_vector: int,
        ae_type: Type[AutoEncoderBase] = AutoEncoder) -> Propagator:

    tcach_lstm = TrainCache.load(project_name, LSTM)
    embed_rule = create_default_embedding_rule(project_name, n_angle_vector, ae_type=ae_type)
    assert tcach_lstm.best_model is not None
    propagator = Propagator(tcach_lstm.best_model, embed_rule)
    return propagator
