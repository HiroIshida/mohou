try:
    from moviepy.editor import ImageSequenceClip
except Exception:
    ImageSequenceClip = None

import argparse
from pathlib import Path
from typing import Union

from mohou.dataset.chimera_dataset import ChimeraDataset
from mohou.dataset.sequence_dataset import AutoRegressiveDatasetConfig
from mohou.encoder import ImageEncoder
from mohou.encoding_rule import EncodingRule
from mohou.file import get_project_path
from mohou.model import LSTM, LSTMConfig, VariationalAutoEncoder
from mohou.model.chimera import Chimera, ChimeraConfig
from mohou.script_utils import create_default_logger
from mohou.setting import setting
from mohou.trainer import TrainCache, TrainConfig, train
from mohou.types import EpisodeBundle

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-pn", type=str, default=setting.primary_project_name, help="project name")
    parser.add_argument("-n", type=int, default=3000, help="iteration number")
    parser.add_argument("-aug", type=int, default=4, help="number of augmentation X")
    parser.add_argument("-hidden", type=int, default=200, help="number of hidden state of lstm")
    parser.add_argument("-layer", type=int, default=2, help="number of layers of lstm")
    parser.add_argument("-rate", type=float, default=2e-4, help="learning rate")
    parser.add_argument(
        "-valid-ratio", type=float, default=0.1, help="split rate for validation dataset"
    )
    parser.add_argument("--pretrained_lstm", action="store_true", help="use pretrained lstm")
    args = parser.parse_args()

    project_name: str = args.pn
    n_epoch: int = args.n
    n_aug: int = args.aug
    n_hidden: int = args.hidden
    n_layer: int = args.layer
    learning_rate: float = args.rate
    valid_ratio: float = args.valid_ratio
    use_pretrained_lstm: bool = args.pretrained_lstm

    assert use_pretrained_lstm

    project_path = get_project_path(project_name)

    logger = create_default_logger(project_path, "chimera")  # noqa
    train_config = TrainConfig(
        n_epoch=n_epoch, batch_size=5, valid_data_ratio=valid_ratio, learning_rate=learning_rate
    )

    bundle = EpisodeBundle.load(project_path)

    # balancer is loaded in chimera model. so no balancer is required.
    use_balancer = not use_pretrained_lstm
    encoding_rule_except_image = EncodingRule.create_default(
        project_path,
        include_image_encoder=False,
        use_balancer=use_balancer,
    )

    ae_tcache = TrainCache.load(project_path, VariationalAutoEncoder)
    ae_config = ae_tcache.cache_path(project_path)
    assert ae_config is not None

    if use_pretrained_lstm:
        tcache_lstm = TrainCache.load(project_path, LSTM)
        assert tcache_lstm.cache_path is not None
        lstm_config: Union[Path, LSTMConfig] = tcache_lstm.cache_path(project_path)
    else:
        image_encoder = ImageEncoder.create_default(project_path)
        lstm_dim = encoding_rule_except_image.dimension + image_encoder.output_size
        lstm_config = LSTMConfig(lstm_dim, n_hidden=n_hidden, n_layer=n_layer)

    dataset_config = AutoRegressiveDatasetConfig(n_aug=n_aug)
    dataset = ChimeraDataset.from_bundle(bundle, encoding_rule_except_image, config=dataset_config)

    conf = ChimeraConfig(lstm_config=lstm_config, ae_config=ae_config)
    model = Chimera(conf)  # type: ignore[var-annotated]
    tcache_chimera = TrainCache.from_model(model)  # type: ignore[var-annotated]
    train(project_path, tcache_chimera, dataset, train_config)
