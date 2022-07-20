import argparse

from mohou.default import create_default_encoding_rule
from mohou.file import get_project_path
from mohou.model import LSTMConfig
from mohou.script_utils import create_default_logger, train_chimera
from mohou.setting import setting
from mohou.trainer import TrainConfig

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-pn", type=str, default=setting.primary_project_name, help="project name")
    parser.add_argument("-n", type=int, default=3000, help="iteration number")
    parser.add_argument("-aug", type=int, default=2, help="number of augmentation X")
    parser.add_argument("-hidden", type=int, default=200, help="number of hidden state of lstm")
    parser.add_argument("-layer", type=int, default=4, help="number of layers of lstm")
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
    encoding_rule = create_default_encoding_rule(project_path)
    lstm_config = LSTMConfig(encoding_rule.dimension, n_hidden=n_hidden, n_layer=n_layer)
    train_config = TrainConfig(n_epoch=n_epoch, batch_size=30, valid_data_ratio=valid_ratio)

    train_chimera(project_path, encoding_rule, lstm_config, train_config)
