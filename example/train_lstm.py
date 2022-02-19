from mohou.trainer import TrainCache

from mohou.types import MultiEpisodeChunk
from mohou.types import AngleVector

from mohou.model import AutoEncoder
from mohou.dataset import AutoRegressiveDataset
from mohou.embedding_functor import IdenticalEmbeddingFunctor
from mohou.embedding_rule import RGBAngelVectorEmbeddingRule

project_name = 'kuka_reaching'
chunk = MultiEpisodeChunk.load(project_name)

tcache_autoencoder = TrainCache.load(project_name, AutoEncoder)
image_embed_func = tcache_autoencoder.best_model.get_embedding_functor()

av_idendical_func = IdenticalEmbeddingFunctor(chunk.get_element_shape(AngleVector)[0])
embed_rule = RGBAngelVectorEmbeddingRule(image_embed_func, av_idendical_func)

dataset = AutoRegressiveDataset.from_chunk(chunk, embed_rule)
