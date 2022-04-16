import argparse
from typing import Type

from mohou.model.lstm import LSTMConfig
from mohou.trainer import TrainCache, TrainConfig, train
from mohou.types import MultiEpisodeChunk
from mohou.types import AngleVector, ImageBase, get_element_type
from mohou.model import AutoEncoder, LSTM
from mohou.dataset import AutoRegressiveDataset
from mohou.dataset import AutoRegressiveDatasetConfig
from mohou.embedder import IdenticalEmbedder
from mohou.embedding_rule import EmbeddingRule
from mohou.utils import create_default_logger

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-pn', type=str, default='kuka_reaching', help='project name')
    parser.add_argument('-n', type=int, default=3000, help='iteration number')
    parser.add_argument('-aug', type=int, default=2, help='number of augmentation X')
    parser.add_argument('-cov-scale', type=float, default=0.1, help='covariance scale in aug')
    parser.add_argument('-image', type=str, default='RGBImage', help='image type')
    parser.add_argument('-valid-ratio', type=float, default=0.1, help='split rate for validation dataset')
    parser.add_argument('-timer-period', type=int, default=10, help='timer period')
    args = parser.parse_args()

    project_name = args.pn
    n_epoch = args.n
    n_aug = args.aug
    cov_scale = args.cov_scale
    valid_ratio = args.valid_ratio
    timer_period = args.timer_period
    image_type: Type[ImageBase] = get_element_type(args.image)  # type: ignore

    logger = create_default_logger(project_name, 'lstm')

    chunk = MultiEpisodeChunk.load(project_name)

    tcache_autoencoder = TrainCache.load(project_name, AutoEncoder)
    image_embed_func = tcache_autoencoder.best_model.get_embedder()

    av_idendical_func = IdenticalEmbedder(AngleVector, chunk.get_element_shape(AngleVector)[0])
    embed_rule = EmbeddingRule.from_embedders([image_embed_func, av_idendical_func])

    dsconfig = AutoRegressiveDatasetConfig(n_aug, cov_scale=cov_scale)
    dataset = AutoRegressiveDataset.from_chunk(chunk, embed_rule, dsconfig)

    lstm_model = LSTM(LSTMConfig(embed_rule.dimension))

    tconfig = TrainConfig(n_epoch=n_epoch, valid_data_ratio=valid_ratio)
    tcache = TrainCache(project_name, timer_period=timer_period)  # type: ignore[var-annotated]
    train(lstm_model, dataset, tcache, config=tconfig)
