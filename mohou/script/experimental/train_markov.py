import argparse
from enum import Enum
from pathlib import Path
from typing import Optional, Type

from mohou.dataset.sequence_dataset import (
    AutoRegressiveDataset,
    AutoRegressiveDatasetConfig,
)
from mohou.encoding_rule import EncodingRule
from mohou.file import get_project_path
from mohou.model.common import ModelBase, ModelConfigBase, ModelT
from mohou.model.experimental import (
    MarkoveModelConfig,
    MarkovPredictionModel,
    ProportionalModel,
    ProportionalModelConfig,
)
from mohou.script_utils import create_default_logger
from mohou.setting import setting
from mohou.trainer import TrainCache, TrainConfig, train
from mohou.types import EpisodeBundle


class AcceptableModel(Enum):
    proportional = ProportionalModel
    markov_prediction = MarkovPredictionModel


def train_proportional(
    project_path: Path,
    model: ModelT,
    encoding_rule: EncodingRule,
    dataset_config: AutoRegressiveDatasetConfig,
    train_config: TrainConfig,
) -> TrainCache[ModelT]:

    bundle = EpisodeBundle.load(project_path)

    dataset = AutoRegressiveDataset.from_bundle(
        bundle,
        encoding_rule,
        dataset_config=dataset_config,
    )
    tcache = TrainCache.from_model(model)  # type: ignore
    train(project_path, tcache, dataset, config=train_config)
    return tcache


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-pn", type=str, default=setting.primary_project_name, help="project name")
    parser.add_argument("-pp", type=str, help="project path. preferred over pn.")
    parser.add_argument("-n", type=int, default=10000, help="iteration number")
    parser.add_argument("-aug", type=int, default=9, help="number of augmentation X")
    parser.add_argument("-model", type=str, default="markov_prediction", help="model name")
    parser.add_argument("-cov-scale", type=float, default=0.1, help="covariance scale in aug")
    parser.add_argument(
        "-valid-ratio", type=float, default=0.1, help="split rate for validation dataset"
    )
    args = parser.parse_args()

    project_name: str = args.pn
    project_path_str: Optional[str] = args.pp
    n_epoch: int = args.n
    n_aug: int = args.aug
    model_name_str: str = args.model
    cov_scale: float = args.cov_scale
    valid_ratio: float = args.valid_ratio

    if project_path_str is None:
        project_path = get_project_path(project_name)
    else:
        project_path = Path(project_path_str)

    model_type: Type[ModelBase] = AcceptableModel[model_name_str].value

    logger = create_default_logger(project_path, model_name_str)

    encoding_rule = EncodingRule.create_default(project_path)
    if model_type == ProportionalModel:
        model_config: ModelConfigBase = ProportionalModelConfig(encoding_rule.dimension, n_layer=3)
    elif model_type == MarkovPredictionModel:
        model_config = MarkoveModelConfig(
            encoding_rule.dimension,
            encoding_rule.dimension,
            n_hidden=128,
            n_layer=4,
            activation="relu",
        )
    else:
        assert False
    model: ModelBase = model_type(model_config)

    dataset_config = AutoRegressiveDatasetConfig(n_aug=n_aug, cov_scale=cov_scale)
    train_config = TrainConfig(n_epoch=n_epoch, valid_data_ratio=valid_ratio)

    train_proportional(
        project_path,
        model,
        encoding_rule,
        dataset_config,
        train_config,
    )
