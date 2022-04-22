from mohou.trainer import TrainCache
from mohou.model import AutoEncoder, LSTM
from mohou.types import AngleVector, TerminateFlag
from mohou.embedder import IdenticalEmbedder
from mohou.embedding_rule import EmbeddingRule
from mohou.propagator import Propagator


def create_default_embedding_rule(project_name: str, n_angle_vector: int) -> EmbeddingRule:
    tcache_autoencoder = TrainCache.load(project_name, AutoEncoder)

    image_embed_func = tcache_autoencoder.best_model.get_embedder()
    av_idendical_func = IdenticalEmbedder(AngleVector, n_angle_vector)
    ef_identical_func = IdenticalEmbedder(TerminateFlag, 1)
    embed_rule = EmbeddingRule.from_embedders(
        [image_embed_func, av_idendical_func, ef_identical_func])
    return embed_rule


def create_default_propagator(project_name: str, n_angle_vector: int) -> Propagator:
    tcach_lstm = TrainCache.load(project_name, LSTM)
    embed_rule = create_default_embedding_rule(project_name, n_angle_vector)
    propagator = Propagator(tcach_lstm.best_model, embed_rule)
    return propagator
