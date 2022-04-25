import os
import re
import random
import pickle
import time
import logging
from logging import Logger
from typing import Type
import matplotlib.pyplot as plt

from mohou.types import ImageBase
from mohou.types import MultiEpisodeChunk
from mohou.model import AutoEncoder
from mohou.model import AutoEncoderConfig
from mohou.model import LSTM
from mohou.model import LSTMConfig
from mohou.dataset import AutoEncoderDataset
from mohou.dataset import AutoEncoderDatasetConfig
from mohou.dataset import AutoRegressiveDataset
from mohou.dataset import AutoRegressiveDatasetConfig
from mohou.file import get_project_dir, get_subproject_dir
from mohou.trainer import TrainCache, TrainConfig, train
from mohou.embedding_rule import EmbeddingRule


def create_default_logger(project_name: str, prefix: str) -> Logger:
    timestr = "_" + time.strftime("%Y%m%d%H%M%S")
    log_dir = os.path.join(get_project_dir(project_name), 'log')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_file_name = os.path.join(log_dir, (prefix + timestr + '.log'))
    FORMAT = '[%(levelname)s] %(asctime)s %(name)s: %(message)s'
    logging.basicConfig(filename=log_file_name, format=FORMAT)
    logger = logging.getLogger('mohou')
    logger.setLevel(level=logging.INFO)

    log_sym_name = os.path.join(log_dir, ('latest_' + prefix + '.log'))
    logger.info('create log symlink :{0} => {1}'.format(log_file_name, log_sym_name))
    if os.path.islink(log_sym_name):
        os.unlink(log_sym_name)
    os.symlink(log_file_name, log_sym_name)
    return logger


def train_autoencoder(
        project_name: str,
        image_type: Type[ImageBase],
        use_aux_data: bool,
        model_config: AutoEncoderConfig,
        dataset_config: AutoEncoderDatasetConfig,
        train_config: TrainConfig):

    logger = create_default_logger(project_name, 'autoencoder')

    chunk = MultiEpisodeChunk.load(project_name)

    if use_aux_data:
        chunk_aux = MultiEpisodeChunk.load_aux(project_name)
        chunk.merge(chunk_aux)
        logger.info('aux data found and merged')

    dataset = AutoEncoderDataset.from_chunk(chunk, image_type, dataset_config)
    tcache = TrainCache(project_name)  # type: ignore[var-annotated]
    model = AutoEncoder(model_config)  # type: ignore
    train(model, dataset, tcache, config=train_config)


def train_lstm(
        project_name: str,
        embedding_rule: EmbeddingRule,
        model_config: LSTMConfig,
        dataset_config: AutoRegressiveDatasetConfig,
        train_config: TrainConfig):

    logger = create_default_logger(project_name, 'lstm')  # noqa

    chunk = MultiEpisodeChunk.load(project_name)
    dataset = AutoRegressiveDataset.from_chunk(chunk, embedding_rule, dataset_config)
    lstm_model = LSTM(model_config)
    tcache = TrainCache(project_name)  # type: ignore[var-annotated]
    train(lstm_model, dataset, tcache, config=train_config)


def visualize_train_histories(project_name: str):
    project_dir = get_project_dir(project_name)
    fnames = os.listdir(project_dir)

    plot_dir = get_subproject_dir(project_name, 'train_history')
    for fname in fnames:
        m = re.match(r'.*TrainCache.*', fname)
        if m is not None:
            pickle_file = os.path.join(project_dir, fname)

            with open(pickle_file, 'rb') as f:
                tcache: TrainCache = pickle.load(f)
                fig, ax = plt.subplots()
                tcache.visualize((fig, ax))
                image_file = os.path.join(plot_dir, fname + '.png')
                fig.savefig(image_file)
                print('saved to {}'.format(image_file))


def visualize_image_reconstruction(
        project_name: str, image_type: Type[ImageBase], n_vis: int = 5):

    chunk = MultiEpisodeChunk.load(project_name)
    chunk_intact = chunk.get_intact_chunk()
    chunk_not_intact = chunk.get_not_intact_chunk()

    no_aug = AutoEncoderDatasetConfig(0)  # to feed not randomized image
    dataset_intact = AutoEncoderDataset.from_chunk(chunk_intact, image_type, no_aug)
    dataset_not_intact = AutoEncoderDataset.from_chunk(chunk_not_intact, image_type, no_aug)

    tcache = TrainCache.load(project_name, AutoEncoder)

    for dataset, postfix in zip([dataset_intact, dataset_not_intact], ['intact', 'not_intact']):
        idxes = list(range(len(dataset)))
        random.shuffle(idxes)
        idxes_test = idxes[:min(n_vis, len(dataset))]

        for i, idx in enumerate(idxes_test):

            image_torch = dataset[idx].unsqueeze(dim=0)
            image_torch_reconstructed = tcache.best_model(image_torch)

            img = dataset.image_type.from_tensor(image_torch.squeeze(dim=0))
            img_reconstructed = dataset.image_type.from_tensor(image_torch_reconstructed.squeeze(dim=0))

            fig, (ax1, ax2) = plt.subplots(1, 2)
            fig.suptitle('left: original, right: reconstructed')
            ax1.imshow(img.to_rgb()._data)
            ax2.imshow(img_reconstructed.to_rgb()._data)
            save_dir = get_subproject_dir(project_name, 'autoencoder_result')

            full_file_name = os.path.join(save_dir, 'result-{}-{}.png'.format(postfix, i))
            plt.savefig(full_file_name)
