import argparse
from pathlib import Path
from typing import Optional, Type

from mohou.dataset import AutoEncoderDatasetConfig
from mohou.file import get_project_path
from mohou.model import AutoEncoder, AutoEncoderBase, VariationalAutoEncoder
from mohou.model.autoencoder import AutoEncoderConfig
from mohou.script_utils import create_default_logger, train_autoencoder
from mohou.setting import setting
from mohou.trainer import TrainConfig
from mohou.types import EpisodeBundle, ImageBase, RGBImage, get_element_type

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-pn", type=str, default=setting.primary_project_name, help="project name")
    parser.add_argument("-pp", type=str, help="project path. preferred over pn")
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

    project_name: str = args.pn
    project_path_str: Optional[str] = args.pp
    n_epoch: int = args.n
    n_aug: int = args.aug
    n_bottleneck: int = args.latent
    valid_ratio: float = args.valid_ratio
    use_vae: bool = args.vae
    warm_start: bool = args.warm
    bundle_postfix: Optional[str] = args.bundle_postfix

    if project_path_str is None:
        project_path = get_project_path(project_name)
    else:
        project_path = Path(project_path_str)

    logger = create_default_logger(project_path, "autoencoder")

    if bundle_postfix == "":
        bundle_postfix = None
    bundle = EpisodeBundle.load(project_path, bundle_postfix)

    image_type: Type[ImageBase] = get_element_type(args.image)  # type: ignore
    n_pixel, _, _ = bundle.spec.type_shape_table[RGBImage]  # Assuming bundle contains rgb
    model_config = AutoEncoderConfig(image_type, n_bottleneck, n_pixel)
    dataset_config = AutoEncoderDatasetConfig(n_aug)
    train_config = TrainConfig(n_epoch=n_epoch, valid_data_ratio=valid_ratio)
    ae_type: Type[AutoEncoderBase] = VariationalAutoEncoder if use_vae else AutoEncoder  # type: ignore
    train_autoencoder(
        project_path,
        image_type,
        model_config,
        dataset_config,
        train_config,
        bundle=bundle,
        ae_type=ae_type,
        warm_start=warm_start,
    )
