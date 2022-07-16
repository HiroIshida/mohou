import argparse
from typing import Type

from mohou.dataset import AutoEncoderDatasetConfig
from mohou.model import AutoEncoder, AutoEncoderBase, VariationalAutoEncoder
from mohou.model.autoencoder import AutoEncoderConfig
from mohou.script_utils import create_default_logger, train_autoencoder
from mohou.setting import setting
from mohou.trainer import TrainConfig
from mohou.types import EpisodeBundle, ImageBase, RGBImage, get_element_type

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-pn", type=str, default=setting.primary_project_name, help="project name")
    parser.add_argument("-n", type=int, default=3000, help="iteration number")
    parser.add_argument("-aug", type=int, default=2, help="number of augmentation X")
    parser.add_argument("-latent", type=int, default=16, help="latent space dim")
    parser.add_argument("-image", type=str, default="RGBImage", help="image type")
    parser.add_argument("-bundle_postfix", type=str, default="", help="postfix for bundle")
    parser.add_argument(
        "-valid-ratio", type=float, default=0.1, help="split rate for validation dataset"
    )
    parser.add_argument("--vae", action="store_true", help="use vae")
    parser.add_argument("--warm", action="store_true", help="warm start")
    args = parser.parse_args()

    project_name = args.pn
    n_epoch = args.n
    n_aug = args.aug
    n_bottleneck = args.latent
    valid_ratio = args.valid_ratio
    use_vae = args.vae
    warm_start = args.warm
    bundle_postfix = args.bundle_postfix

    logger = create_default_logger(project_name, "autoencoder")

    if bundle_postfix == "":
        bundle_postfix = None
    bundle = EpisodeBundle.load(project_name, bundle_postfix)

    image_type: Type[ImageBase] = get_element_type(args.image)  # type: ignore
    n_pixel, _, _ = bundle.spec.type_shape_table[RGBImage]  # Assuming bundle contains rgb
    model_config = AutoEncoderConfig(image_type, n_bottleneck, n_pixel)
    dataset_config = AutoEncoderDatasetConfig(n_aug)
    train_config = TrainConfig(n_epoch=n_epoch, valid_data_ratio=valid_ratio)
    ae_type: Type[AutoEncoderBase] = VariationalAutoEncoder if use_vae else AutoEncoder  # type: ignore
    train_autoencoder(
        project_name,
        image_type,
        model_config,
        dataset_config,
        train_config,
        bundle=bundle,
        ae_type=ae_type,
        warm_start=warm_start,
    )
