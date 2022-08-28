from pathlib import Path
from typing import Optional

try:
    from moviepy.editor import ImageSequenceClip
except Exception:
    ImageSequenceClip = None

import argparse

from mohou.dataset.chimera_dataset import ChimeraDataset
from mohou.dataset.sequence_dataset import AutoRegressiveDatasetConfig
from mohou.default import create_default_encoding_rule, load_default_image_encoder
from mohou.encoding_rule import EncodingRule
from mohou.file import get_project_path
from mohou.model import AutoEncoder, LSTMConfig
from mohou.model.chimera import Chimera, ChimeraConfig
from mohou.script_utils import create_default_logger
from mohou.setting import setting
from mohou.trainer import TrainCache, TrainConfig, train
from mohou.types import EpisodeBundle


def train_chimera(
    project_path: Path,
    encoding_rule: EncodingRule,
    lstm_config: LSTMConfig,
    train_config: TrainConfig,
    n_aug: int = 4,
    bundle: Optional[EpisodeBundle] = None,
):  # TODO(HiroIshida): minimal args
    # TODO(HiroIshida): maybe better to move this function to script_utils

    if bundle is None:
        bundle = EpisodeBundle.load(project_path)

    dataset_config = AutoRegressiveDatasetConfig(n_aug=n_aug)
    dataset = ChimeraDataset.from_bundle(bundle, encoding_rule, dataset_config=dataset_config)
    ae = TrainCache.load(project_path, AutoEncoder).best_model
    conf = ChimeraConfig(lstm_config, ae_config=ae)
    model = Chimera(conf)  # type: ignore[var-annotated]
    tcache = TrainCache.from_model(model)  # type: ignore[var-annotated]
    train(project_path, tcache, dataset, train_config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-pn", type=str, default=setting.primary_project_name, help="project name")
    parser.add_argument("-n", type=int, default=3000, help="iteration number")
    parser.add_argument("-aug", type=int, default=4, help="number of augmentation X")
    parser.add_argument("-hidden", type=int, default=200, help="number of hidden state of lstm")
    parser.add_argument("-layer", type=int, default=2, help="number of layers of lstm")
    parser.add_argument(
        "-valid-ratio", type=float, default=0.1, help="split rate for validation dataset"
    )
    args = parser.parse_args()

    project_name: str = args.pn
    n_epoch: int = args.n
    n_aug: int = args.aug
    n_hidden: int = args.hidden
    n_layer: int = args.layer
    valid_ratio: float = args.valid_ratio

    project_path = get_project_path(project_name)

    logger = create_default_logger(project_path, "chimera")  # noqa
    encoding_rule_except_image = create_default_encoding_rule(
        project_path, include_image_encoder=False
    )
    image_encoder = load_default_image_encoder(project_path)
    lstm_dim = encoding_rule_except_image.dimension + image_encoder.output_size

    lstm_config = LSTMConfig(lstm_dim, n_hidden=n_hidden, n_layer=n_layer)
    train_config = TrainConfig(n_epoch=n_epoch, batch_size=5, valid_data_ratio=valid_ratio)

    train_chimera(project_path, encoding_rule_except_image, lstm_config, train_config, n_aug=n_aug)
