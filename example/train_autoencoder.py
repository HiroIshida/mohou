import argparse

import torch

from mohou.dataset import AutoEncoderDataset
from mohou.model import AutoEncoder
from mohou.model.autoencoder import AutoEncoderConfig
from mohou.trainer import TrainCache, TrainConfig, train
from mohou.types import MultiEpisodeChunk
from mohou.utils import split_with_ratio

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-pn', type=str, default='kuka_reaching', help='project name')
    parser.add_argument('-n', type=int, default=3000, help='iteration number')
    parser.add_argument('-valid-ratio', type=float, default=0.1, help='split rate for validation dataset')
    args = parser.parse_args()

    project_name = args.pn
    n_epoch = args.n
    valid_ratio = args.valid_ratio

    chunk = MultiEpisodeChunk.load(project_name)
    dataset = AutoEncoderDataset.from_chunk(chunk)
    dataset_train, dataset_valid = split_with_ratio(dataset, valid_ratio=valid_ratio)

    tcache = TrainCache[AutoEncoder](project_name)
    model = AutoEncoder(torch.device('cpu'), AutoEncoderConfig())
    tconfig = TrainConfig(n_epoch=n_epoch)
    train(model, dataset_train, dataset_valid, tcache, config=tconfig)
