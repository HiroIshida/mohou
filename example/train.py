import torch

from mohou.dataset import AutoEncoderDataset
from mohou.model import AutoEncoder
from mohou.model.autoencoder import AutoEncoderConfig
from mohou.trainer import TrainCache, TrainConfig, train
from mohou.types import MultiEpisodeChunk
from mohou.utils import split_with_ratio

project_name = 'kuka_reaching'
chunk = MultiEpisodeChunk.load(project_name)
dataset = AutoEncoderDataset.from_chunk(chunk)
dataset_train, dataset_valid = split_with_ratio(dataset)

tcache = TrainCache[AutoEncoder](project_name)
model = AutoEncoder(torch.device('cpu'), AutoEncoderConfig())
tconfig = TrainConfig(n_epoch=3)
train(model, dataset_train, dataset_valid, tcache, config=tconfig)
