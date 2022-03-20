import argparse
from typing import Type

from mohou.dataset import AutoEncoderDataset
from mohou.model import AutoEncoder
from mohou.model.autoencoder import AutoEncoderConfig
from mohou.trainer import TrainCache, TrainConfig, train
from mohou.types import ImageBase, MultiEpisodeChunk, get_element_type
from mohou.utils import create_default_logger

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-pn', type=str, default='kuka_reaching', help='project name')
    parser.add_argument('-n', type=int, default=3000, help='iteration number')
    parser.add_argument('-m', type=int, default=112, help='pixel number')
    parser.add_argument('-image', type=str, default='RGBImage', help='image type')
    parser.add_argument('-valid-ratio', type=float, default=0.1, help='split rate for validation dataset')
    parser.add_argument('-timer-period', type=int, default=10, help='timer period')
    args = parser.parse_args()

    project_name = args.pn
    n_epoch = args.n
    n_pixel = args.m
    valid_ratio = args.valid_ratio
    timer_period = args.timer_period
    image_type: Type[ImageBase] = get_element_type(args.image)  # type: ignore

    logger = create_default_logger(project_name, 'autoencoder')

    chunk = MultiEpisodeChunk.load(project_name)
    dataset = AutoEncoderDataset.from_chunk(chunk, image_type)

    tcache = TrainCache(project_name, timer_period=timer_period)  # type: ignore[var-annotated]
    model = AutoEncoder(AutoEncoderConfig(image_type=image_type, n_pixel=n_pixel))  # type: ignore
    tconfig = TrainConfig(n_epoch=n_epoch, valid_data_ratio=valid_ratio)
    train(model, dataset, tcache, config=tconfig)
