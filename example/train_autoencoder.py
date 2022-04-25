import argparse
from typing import Type

from mohou.dataset import AutoEncoderDatasetConfig
from mohou.model.autoencoder import AutoEncoderConfig
from mohou.trainer import TrainConfig
from mohou.types import RGBImage, ImageBase, MultiEpisodeChunk, get_element_type
from mohou.script_utils import train_autoencoder

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-pn', type=str, default='kuka_reaching', help='project name')
    parser.add_argument('-n', type=int, default=3000, help='iteration number')
    parser.add_argument('-aug', type=int, default=2, help='number of augmentation X')
    parser.add_argument('-latent', type=int, default=16, help='latent space dim')
    parser.add_argument('-image', type=str, default='RGBImage', help='image type')
    parser.add_argument('-valid-ratio', type=float, default=0.1, help='split rate for validation dataset')
    parser.add_argument('--aux', action='store_true', help='use auxiliary data')
    args = parser.parse_args()

    project_name = args.pn
    n_epoch = args.n
    n_aug = args.aug
    n_bottleneck = args.latent
    valid_ratio = args.valid_ratio
    use_aux_data = args.aux

    image_type: Type[ImageBase] = get_element_type(args.image)  # type: ignore
    chunk_spec = MultiEpisodeChunk.load_spec(project_name)
    n_pixel, _, _ = chunk_spec.type_shape_table[RGBImage]  # Assuming chunk contains rgb
    model_config = AutoEncoderConfig(image_type, n_bottleneck, n_pixel)
    dataset_config = AutoEncoderDatasetConfig(n_aug)
    train_config = TrainConfig(n_epoch=n_epoch, valid_data_ratio=valid_ratio)
    train_autoencoder(project_name, image_type, use_aux_data, model_config, dataset_config, train_config)
