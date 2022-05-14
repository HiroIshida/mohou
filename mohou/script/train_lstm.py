import argparse

from mohou.model.lstm import LSTMConfig
from mohou.trainer import TrainConfig
from mohou.dataset import AutoRegressiveDatasetConfig
from mohou.default import create_default_embedding_rule
from mohou.script_utils import train_lstm, create_default_logger

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-pn', type=str, default='kuka_reaching', help='project name')
    parser.add_argument('-n', type=int, default=3000, help='iteration number')
    parser.add_argument('-aug', type=int, default=2, help='number of augmentation X')
    parser.add_argument('-cov-scale', type=float, default=0.1, help='covariance scale in aug')
    parser.add_argument('-valid-ratio', type=float, default=0.1, help='split rate for validation dataset')
    parser.add_argument('--warm', action='store_true', help='warm start')
    args = parser.parse_args()

    project_name = args.pn
    n_epoch = args.n
    n_aug = args.aug
    cov_scale = args.cov_scale
    valid_ratio = args.valid_ratio
    warm_start = args.warm

    logger = create_default_logger(project_name, 'lstm')  # noqa

    embedding_rule = create_default_embedding_rule(project_name)
    model_config = LSTMConfig(embedding_rule.dimension)
    dataset_config = AutoRegressiveDatasetConfig(n_aug, cov_scale=cov_scale)
    train_config = TrainConfig(n_epoch=n_epoch, valid_data_ratio=valid_ratio)
    train_lstm(project_name, embedding_rule, model_config, dataset_config, train_config, warm_start=warm_start)
