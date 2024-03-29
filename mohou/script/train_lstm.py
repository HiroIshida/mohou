import argparse
from pathlib import Path
from typing import List, Optional

import numpy as np

from mohou.dataset import AutoRegressiveDatasetConfig
from mohou.encoder import ImageEncoder
from mohou.encoding_rule import EncodingRule
from mohou.file import get_project_path
from mohou.model.lstm import LSTM, LSTMConfig
from mohou.script_utils import create_default_logger, train_lstm
from mohou.setting import setting
from mohou.trainer import TrainCache, TrainConfig
from mohou.types import EpisodeBundle


def create_default_image_context_list(
    project_path: Path, bundle: Optional[EpisodeBundle] = None
) -> List[np.ndarray]:
    if bundle is None:
        bundle = EpisodeBundle.load(project_path)
    image_encoder = ImageEncoder.create_default(project_path)

    context_list = []
    for episode in bundle:
        seq = episode.get_sequence_by_type(image_encoder.elem_type)
        context = image_encoder.forward(seq[0])
        context_list.append(context)

    return context_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-pn", type=str, default=setting.primary_project_name, help="project name")
    parser.add_argument("-pp", type=str, help="project path. preferred over pn.")
    parser.add_argument("-n", type=int, default=3000, help="iteration number")
    parser.add_argument("-aug", type=int, default=2, help="number of augmentation X")
    parser.add_argument("-hidden", type=int, default=200, help="number of hidden state of lstm")
    parser.add_argument(
        "-window",
        type=int,
        help="window size of episode cutting. If None, episode will not be cut.",
    )
    parser.add_argument("-layer", type=int, default=4, help="number of layers of lstm")
    parser.add_argument("-cov-scale", type=float, default=0.1, help="covariance scale in aug")
    parser.add_argument(
        "-avbias-std", type=float, default=0.0, help="std of angle vector calibration bias std"
    )
    parser.add_argument(
        "-valid-ratio", type=float, default=0.1, help="split rate for validation dataset"
    )
    parser.add_argument("--variational", action="store_true", help="variational")
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
    window_size: Optional[int] = args.window
    cov_scale: float = args.cov_scale
    av_calibration_bias_std: float = args.avbias_std
    valid_ratio: float = args.valid_ratio
    warm_start: bool = args.warm
    variational_lstm: bool = args.variational
    use_image_context: bool = args.use_image_context

    if project_path_str is None:
        project_path = get_project_path(project_name)
    else:
        project_path = Path(project_path_str)

    logger = create_default_logger(project_path, "lstm")  # noqa

    encoding_rule = EncodingRule.create_default(project_path)
    model_config = LSTMConfig(
        encoding_rule.dimension,
        n_hidden=n_hidden,
        n_layer=n_layer,
        variational=variational_lstm,
        type_wise_loss=True,
        type_bound_table=encoding_rule.type_bound_table,
    )
    dataset_config = AutoRegressiveDatasetConfig(
        n_aug=n_aug,
        cov_scale=cov_scale,
        window_size=window_size,
        av_calibration_bias_std=av_calibration_bias_std,
    )
    train_config = TrainConfig(n_epoch=n_epoch, valid_data_ratio=valid_ratio)

    context_list = None
    if use_image_context:
        context_list = create_default_image_context_list(project_path)
        model_config.n_static_context = len(context_list[0])

    if warm_start:
        tcache_pretrained: Optional[TrainCache[LSTM]] = TrainCache.load_latest(project_path, LSTM)
    else:
        tcache_pretrained = None

    train_lstm(
        project_path,
        encoding_rule,
        model_config,
        dataset_config,
        train_config,
        tcache_pretrained=tcache_pretrained,
        context_list=context_list,
    )
