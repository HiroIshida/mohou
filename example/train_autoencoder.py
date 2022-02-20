import argparse

from mohou.dataset import RGBAutoEncoderDataset
from mohou.model import AutoEncoder
from mohou.model.autoencoder import AutoEncoderConfig
from mohou.trainer import TrainCache, TrainConfig, train
from mohou.types import MultiEpisodeChunk
from mohou.utils import create_default_logger, detect_device, split_with_ratio

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-pn', type=str, default='kuka_reaching', help='project name')
    parser.add_argument('-n', type=int, default=3000, help='iteration number')
    parser.add_argument('-valid-ratio', type=float, default=0.1, help='split rate for validation dataset')
    args = parser.parse_args()

    project_name = args.pn
    n_epoch = args.n
    valid_ratio = args.valid_ratio

    logger = create_default_logger(project_name, 'autoencoder')

    chunk = MultiEpisodeChunk.load(project_name)
    dataset = RGBAutoEncoderDataset.from_chunk(chunk)
    dataset_train, dataset_valid = split_with_ratio(dataset, valid_ratio=valid_ratio)

    tcache = TrainCache[AutoEncoder](project_name)
    model = AutoEncoder(detect_device(), AutoEncoderConfig())
    tconfig = TrainConfig(n_epoch=n_epoch)
    train(model, dataset_train, dataset_valid, tcache, config=tconfig)
