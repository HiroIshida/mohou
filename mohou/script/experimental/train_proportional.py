import argparse
from pathlib import Path
from typing import Optional

from mohou.dataset.sequence_dataset import (
    AutoRegressiveDataset,
    AutoRegressiveDatasetConfig,
)
from mohou.encoding_rule import EncodingRule
from mohou.file import get_project_path
from mohou.model.experimental import ProportionalModel, ProportionalModelConfig
from mohou.script_utils import create_default_logger
from mohou.setting import setting
from mohou.trainer import TrainCache, TrainConfig, train
from mohou.types import EpisodeBundle


def train_proportional(
    project_path: Path,
    encoding_rule: EncodingRule,
    model_config: ProportionalModelConfig,
    dataset_config: AutoRegressiveDatasetConfig,
    train_config: TrainConfig,
    tcache_pretrained: Optional[TrainCache] = None,
) -> TrainCache[ProportionalModel]:

    bundle = EpisodeBundle.load(project_path)

    dataset = AutoRegressiveDataset.from_bundle(
        bundle,
        encoding_rule,
        dataset_config=dataset_config,
    )

    if tcache_pretrained is not None:
        tcache = tcache_pretrained
    else:
        model = ProportionalModel(model_config)
        tcache = TrainCache.from_model(model)  # type: ignore
    train(project_path, tcache, dataset, config=train_config)
    return tcache


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-pn", type=str, default=setting.primary_project_name, help="project name")
    parser.add_argument("-pp", type=str, help="project path. preferred over pn.")
    parser.add_argument("-n", type=int, default=10000, help="iteration number")
    parser.add_argument("-aug", type=int, default=9, help="number of augmentation X")
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

    if project_path_str is None:
        project_path = get_project_path(project_name)
    else:
        project_path = Path(project_path_str)

    logger = create_default_logger(project_path, "proportional")  # noqa

    encoding_rule = EncodingRule.create_default(project_path)
    model_config = ProportionalModelConfig(encoding_rule.dimension, n_layer=3)
    dataset_config = AutoRegressiveDatasetConfig(n_aug=n_aug, cov_scale=cov_scale)
    train_config = TrainConfig(n_epoch=n_epoch, valid_data_ratio=valid_ratio)

    train_proportional(
        project_path,
        encoding_rule,
        model_config,
        dataset_config,
        train_config,
    )
