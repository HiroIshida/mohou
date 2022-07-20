import argparse
from pathlib import Path
from typing import Optional

from mohou.dataset import AutoRegressiveDatasetConfig
from mohou.default import (
    create_default_encoding_rule,
    create_default_image_context_list,
)
from mohou.file import get_project_path
from mohou.model.lstm import LSTMConfig
from mohou.script_utils import create_default_logger, train_lstm
from mohou.setting import setting
from mohou.trainer import TrainConfig

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-pn", type=str, default=setting.primary_project_name, help="project name")
    parser.add_argument("-pp", type=str, help="project path. preferred over pn.")
    parser.add_argument("-n", type=int, default=3000, help="iteration number")
    parser.add_argument("-aug", type=int, default=2, help="number of augmentation X")
    parser.add_argument("-hidden", type=int, default=200, help="number of hidden state of lstm")
    parser.add_argument("-layer", type=int, default=4, help="number of layers of lstm")
    parser.add_argument("-cov-scale", type=float, default=0.1, help="covariance scale in aug")
    parser.add_argument(
        "-valid-ratio", type=float, default=0.1, help="split rate for validation dataset"
    )
    parser.add_argument("--type_wise_loss", action="store_true", help="use type_wise_loss")
    parser.add_argument("--warm", action="store_true", help="warm start")
    parser.add_argument(
        "--use_image_context", action="store_true", help="initial image as context input"
    )
    args = parser.parse_args()

    project_name: str = args.pn
    project_path_str: Optional[str] = args.pp
    n_epoch: int = args.n
    n_aug: int = args.aug
    n_hidden: int = args.hidden
    n_layer: int = args.layer
    cov_scale: float = args.cov_scale
    valid_ratio: float = args.valid_ratio
    warm_start: bool = args.warm
    use_image_context: bool = args.use_image_context

    if project_path_str is None:
        project_path = get_project_path(project_name)
    else:
        project_path = Path(project_path_str)

    logger = create_default_logger(project_path, "lstm")  # noqa

    encoding_rule = create_default_encoding_rule(project_path)
    model_config = LSTMConfig(
        encoding_rule.dimension,
        n_hidden=n_hidden,
        n_layer=n_layer,
        type_wise_loss=args.type_wise_loss,
        type_bound_table=encoding_rule.type_bound_table,
    )
    dataset_config = AutoRegressiveDatasetConfig(n_aug=n_aug, cov_scale=cov_scale)
    train_config = TrainConfig(n_epoch=n_epoch, valid_data_ratio=valid_ratio)

    context_list = None
    if use_image_context:
        context_list = create_default_image_context_list(project_path)
        model_config.n_static_context = len(context_list[0])

    train_lstm(
        project_path,
        encoding_rule,
        model_config,
        dataset_config,
        train_config,
        warm_start=warm_start,
        context_list=context_list,
    )
