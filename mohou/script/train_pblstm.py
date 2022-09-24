import argparse
from pathlib import Path
from typing import Optional

from mohou.dataset import AutoRegressiveDatasetConfig
from mohou.encoding_rule import EncodingRule
from mohou.file import get_project_path
from mohou.model.lstm import PBLSTM, PBLSTMConfig
from mohou.script_utils import create_default_logger, train_lstm
from mohou.setting import setting
from mohou.trainer import TrainConfig
from mohou.types import EpisodeBundle

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-pn", type=str, default=setting.primary_project_name, help="project name")
    parser.add_argument("-pp", type=str, help="project path. preferred over pn.")
    parser.add_argument("-n", type=int, default=3000, help="iteration number")
    parser.add_argument("-aug", type=int, default=2, help="number of augmentation X")
    parser.add_argument("-hidden", type=int, default=200, help="number of hidden state of lstm")
    parser.add_argument("-layer", type=int, default=4, help="number of layers of lstm")
    parser.add_argument("-cov-scale", type=float, default=0.1, help="covariance scale in aug")
    parser.add_argument("-pb", type=int, default=2, help="parametric bias dimension")
    parser.add_argument(
        "-valid-ratio", type=float, default=0.1, help="split rate for validation dataset"
    )
    parser.add_argument("--warm", action="store_true", help="warm start")
    args = parser.parse_args()

    project_name: str = args.pn
    project_path_str: Optional[str] = args.pp
    n_epoch: int = args.n
    n_aug: int = args.aug
    n_hidden: int = args.hidden
    n_layer: int = args.layer
    n_pb_dim: int = args.pb
    cov_scale: float = args.cov_scale
    valid_ratio: float = args.valid_ratio
    warm_start: bool = args.warm

    if project_path_str is None:
        project_path = get_project_path(project_name)
    else:
        project_path = Path(project_path_str)

    logger = create_default_logger(project_path, "pblstm")  # noqa

    bundle = EpisodeBundle.load(project_path)

    encoding_rule = EncodingRule.create_default(project_path)
    model_config = PBLSTMConfig(
        encoding_rule.dimension,
        n_hidden=n_hidden,
        n_layer=n_layer,
        n_pb_dim=n_pb_dim,
        n_pb=len(bundle),
    )
    dataset_config = AutoRegressiveDatasetConfig(n_aug=n_aug, cov_scale=cov_scale)
    train_config = TrainConfig(n_epoch=n_epoch, valid_data_ratio=valid_ratio)

    train_lstm(
        project_path,
        encoding_rule,
        model_config,
        dataset_config,
        train_config,
        model_type=PBLSTM,
        warm_start=warm_start,
    )
