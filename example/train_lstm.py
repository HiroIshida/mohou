import torch

from mohou.model.lstm import LSTMConfig
from mohou.trainer import TrainCache, TrainConfig, train
from mohou.types import MultiEpisodeChunk
from mohou.types import AngleVector
from mohou.model import AutoEncoder, LSTM
from mohou.dataset import AutoRegressiveDataset
from mohou.embedding_functor import IdenticalEmbeddingFunctor
from mohou.embedding_rule import RGBAngelVectorEmbeddingRule
from mohou.utils import split_with_ratio

project_name = 'kuka_reaching'
chunk = MultiEpisodeChunk.load(project_name)

tcache_autoencoder = TrainCache.load(project_name, AutoEncoder)
image_embed_func = tcache_autoencoder.best_model.get_embedding_functor()

av_idendical_func = IdenticalEmbeddingFunctor(chunk.get_element_shape(AngleVector)[0])
embed_rule = RGBAngelVectorEmbeddingRule(image_embed_func, av_idendical_func)

dataset = AutoRegressiveDataset.from_chunk(chunk, embed_rule)
dataset_train, dataset_valid = split_with_ratio(dataset, valid_ratio=0.5)  # TODO(HiroIshida) fix it

lstm_model = LSTM(torch.device('cpu'), LSTMConfig(embed_rule.dimension))

tconfig = TrainConfig(n_epoch=3)
tcache = TrainCache[LSTM](project_name)
train(lstm_model, dataset_train, dataset_valid, tcache, config=tconfig)
