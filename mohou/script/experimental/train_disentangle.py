import argparse
from pathlib import Path
from typing import Optional

from mohou.dataset.sequence_dataset import (
    AutoRegressiveDataset,
    AutoRegressiveDatasetConfig,
)
from mohou.encoding_rule import EncodingRule
from mohou.file import get_project_path
from mohou.model.experimental import DisentangleLSTM, DisentangleLSTMConfig
from mohou.script_utils import create_default_logger
from mohou.setting import setting
from mohou.trainer import TrainCache, TrainConfig, train
from mohou.types import EpisodeBundle


def train_disentangled_lstm(
    project_path: Path,
    encoding_rule: EncodingRule,
    model_config: DisentangleLSTMConfig,
    dataset_config: AutoRegressiveDatasetConfig,
    train_config: TrainConfig,
) -> TrainCache[DisentangleLSTM]:

    bundle = EpisodeBundle.load(project_path)

    dataset = AutoRegressiveDataset.from_bundle(
        bundle,
        encoding_rule,
        dataset_config=dataset_config,
    )

    model = DisentangleLSTM(model_config)
    tcache = TrainCache.from_model(model)  # type: ignore
    train(project_path, tcache, dataset, config=train_config)
    return tcache


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-pn", type=str, default=setting.primary_project_name, help="project name")
    parser.add_argument("-pp", type=str, help="project path. preferred over pn.")
    parser.add_argument("-n", type=int, default=10000, help="iteration number")
    parser.add_argument("-aug", type=int, default=9, help="number of augmentation X")
    parser.add_argument("-fc_layer", type=int, default=3, help="num of fc layer")
    parser.add_argument("-fc_hidden", type=int, default=100, help="num of fc hidden")
    parser.add_argument("-lstm_layer", type=int, default=1, help="num of lstm layer")
    parser.add_argument("-lstm_hidden", type=int, default=200, help="num of lstm hidden")
    parser.add_argument("-bottleneck", type=int, default=10, help="num of bottleneck dim")
    parser.add_argument("-cov-scale", type=float, default=0.1, help="covariance scale in aug")
    parser.add_argument(
        "-valid-ratio", type=float, default=0.1, help="split rate for validation dataset"
    )
    args = parser.parse_args()

    project_name: str = args.pn
    project_path_str: Optional[str] = args.pp
    n_epoch: int = args.n
    n_aug: int = args.aug
    cov_scale: float = args.cov_scale
    valid_ratio: float = args.valid_ratio
    n_fc_layer: int = args.fc_layer
    n_fc_hidden: int = args.fc_hidden
    n_lstm_layer: int = args.lstm_layer
    n_lstm_hidden: int = args.lstm_hidden
    n_bottleneck: int = args.bottleneck

    if project_path_str is None:
        project_path = get_project_path(project_name)
    else:
        project_path = Path(project_path_str)

    logger = create_default_logger(project_path, "disentangledLstm")  # noqa

    encoding_rule = EncodingRule.create_default(project_path)
    model_config = DisentangleLSTMConfig(
        encoding_rule.dimension,
        n_bottleneck=n_bottleneck,
        n_fc_layer=n_fc_layer,
        n_fc_hidden=n_fc_hidden,
        n_lstm_hidden=n_lstm_hidden,
        n_lstm_layer=n_lstm_layer,
    )
    dataset_config = AutoRegressiveDatasetConfig(n_aug=n_aug, cov_scale=cov_scale)
    train_config = TrainConfig(n_epoch=n_epoch, valid_data_ratio=valid_ratio)

    train_disentangled_lstm(
        project_path,
        encoding_rule,
        model_config,
        dataset_config,
        train_config,
    )
