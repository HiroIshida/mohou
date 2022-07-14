import argparse

from mohou.dataset import MarkovControlSystemDataset
from mohou.encoder import VectorIdenticalEncoder
from mohou.encoding_rule import EncodingRule
from mohou.model import ControlModel, VariationalAutoEncoder
from mohou.model.markov import MarkoveModelConfig
from mohou.script_utils import create_default_logger
from mohou.setting import setting
from mohou.trainer import TrainCache, TrainConfig, train
from mohou.types import AngleVector, EpisodeBundle


def create_obs_rule():
    tcache = TrainCache.load(None, VariationalAutoEncoder)
    model = tcache.best_model
    assert model is not None
    f = model.get_encoder()
    obs_rule = EncodingRule.from_encoders([f])
    return obs_rule


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-pn", type=str, default=setting.primary_project_name, help="project name")
    parser.add_argument("-n", type=int, default=3000, help="iteration number")

    args = parser.parse_args()
    project_name = args.pn
    n_epoch = args.n
    assert project_name is not None

    logger = create_default_logger(project_name, "control_model")

    bundle = EpisodeBundle.load(project_name)
    n_av_dim = bundle.spec.type_shape_table[AngleVector][0]
    f = VectorIdenticalEncoder(AngleVector, n_av_dim)
    ctrl_rule = EncodingRule.from_encoders([f])

    obs_rule = create_obs_rule()

    dataset = MarkovControlSystemDataset.from_bundle(
        bundle, ctrl_rule, obs_rule, diff_as_control=True
    )

    tcache = TrainCache[ControlModel](project_name)
    n_input = ctrl_rule.dimension + obs_rule.dimension
    n_output = obs_rule.dimension
    config = MarkoveModelConfig(n_input, n_output, activation="relu")
    model = ControlModel(config)
    tconfig = TrainConfig(n_epoch=n_epoch)
    train(tcache, dataset, model, config=tconfig)
