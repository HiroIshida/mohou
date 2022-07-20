import argparse
from pathlib import Path

from mohou.dataset import MarkovControlSystemDataset
from mohou.encoder import VectorIdenticalEncoder
from mohou.encoding_rule import EncodingRule
from mohou.file import get_project_path
from mohou.model import ControlModel, VariationalAutoEncoder
from mohou.model.markov import MarkoveModelConfig
from mohou.script_utils import create_default_logger
from mohou.setting import setting
from mohou.trainer import TrainCache, TrainConfig, train
from mohou.types import AngleVector, EpisodeBundle


def create_obs_rule(project_path: Path):
    tcache = TrainCache.load(project_path, VariationalAutoEncoder)
    model = tcache.best_model
    assert model is not None
    f = model.get_encoder()
    obs_rule = EncodingRule.from_encoders([f])
    return obs_rule


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-pn", type=str, default=setting.primary_project_name, help="project name")
    parser.add_argument(
        "-full_path",
        type=str,
        help="project full path. If this arg is set, full path will be prefered over project name",
    )
    parser.add_argument("-n", type=int, default=3000, help="iteration number")

    args = parser.parse_args()
    project_name: str = args.pn
    n_epoch: int = args.n

    if args.full_path is None:
        project_path = get_project_path(project_name)
    else:
        project_path = Path(args.full_path)

    logger = create_default_logger(project_path, "control_model")

    bundle = EpisodeBundle.load(project_path)
    n_av_dim = bundle.spec.type_shape_table[AngleVector][0]
    f = VectorIdenticalEncoder(AngleVector, n_av_dim)
    ctrl_rule = EncodingRule.from_encoders([f])

    obs_rule = create_obs_rule(project_path)

    dataset = MarkovControlSystemDataset.from_bundle(
        bundle, ctrl_rule, obs_rule, diff_as_control=True
    )

    n_input = ctrl_rule.dimension + obs_rule.dimension
    n_output = obs_rule.dimension
    config = MarkoveModelConfig(n_input, n_output, activation="relu")
    tcache = TrainCache[ControlModel].from_model(ControlModel(config))
    tconfig = TrainConfig(n_epoch=n_epoch)
    train(project_path, tcache, dataset, config=tconfig)
