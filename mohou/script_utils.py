import os
import time
import logging
from logging import Logger
from typing import Type

from mohou.types import ImageBase
from mohou.types import MultiEpisodeChunk
from mohou.model import AutoEncoder
from mohou.model import AutoEncoderConfig
from mohou.dataset import AutoEncoderDataset
from mohou.dataset import AutoEncoderDatasetConfig
from mohou.file import get_project_dir
from mohou.trainer import TrainCache, TrainConfig, train


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
        dataset_conf: AutoEncoderDatasetConfig,
        train_conf: TrainConfig):

    logger = create_default_logger(project_name, 'autoencoder')

    chunk = MultiEpisodeChunk.load(project_name)

    if use_aux_data:
        chunk_aux = MultiEpisodeChunk.load_aux(project_name)
        chunk.merge(chunk_aux)
        logger.info('aux data found and merged')

    dataset = AutoEncoderDataset.from_chunk(chunk, image_type, dataset_conf)
    tcache = TrainCache(project_name)  # type: ignore[var-annotated]
    model = AutoEncoder(model_config)  # type: ignore
    train(model, dataset, tcache, config=train_conf)
