import argparse
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np

from mohou.dataset import AutoRegressiveDataset
from mohou.encoder import VectorIdenticalEncoder
from mohou.encoding_rule import EncodingRule
from mohou.file import get_project_path
from mohou.model import LSTM, LSTMConfig
from mohou.propagator import Propagator
from mohou.script_utils import create_default_logger
from mohou.trainer import TrainCache, TrainConfig, train
from mohou.types import (
    AngleVector,
    ElementDict,
    ElementSequence,
    EpisodeBundle,
    EpisodeData,
    TerminateFlag,
)


@dataclass
class SpringMassDumper:
    c: float = 0.08
    k: float = 0.01
    dt: float = 1.0

    def propagate(self, state: np.ndarray) -> np.ndarray:
        x, v, a = state
        x_new = x + v * self.dt
        v_new = v + a * self.dt
        a_new = -self.k * x - self.c * v
        return np.array([x_new, v_new, a_new])

    def sample_random_init_state(self) -> np.ndarray:
        while True:
            x = np.random.randn()
            if np.abs(x) > 1.0:
                break
        return np.array([x, 0.0, 0.0])

    def is_termianate(self, x: np.ndarray) -> bool:
        x, v, a = x
        if np.abs(x) > 0.2:
            return False
        if np.abs(v) > 0.01:
            return False
        if np.abs(a) > 0.001:
            return False
        return True

    def create_bundle(self, n_data: int = 100) -> EpisodeBundle:
        edata_list = []
        for _ in range(n_data):
            state = self.sample_random_init_state()
            av_seq = ElementSequence[AngleVector]([AngleVector(state)])
            while True:
                state = self.propagate(state)
                av_seq.append(AngleVector(state))
                if self.is_termianate(state):
                    break
            edata_list.append(EpisodeData.from_seq_list([av_seq]))
        bundle = EpisodeBundle.from_data_list(edata_list)
        return bundle


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="train instead of using cache")
    args = parser.parse_args()
    with_training: bool = args.train

    project_name = "spring_mass_dumper"
    project_path = get_project_path(project_name)
    create_default_logger(project_path, "LSTM")
    smd = SpringMassDumper()

    av_emb = VectorIdenticalEncoder(AngleVector, 3)
    ef_identical_func = VectorIdenticalEncoder(TerminateFlag, 1)
    rule = EncodingRule.from_encoders([av_emb, ef_identical_func])

    if with_training:
        bundle = smd.create_bundle()
        bundle.dump(project_path)
        dataset = AutoRegressiveDataset.from_bundle(bundle, rule)

        tconfig = TrainConfig(n_epoch=1000)
        mconfig = LSTMConfig(4)
        tcache = TrainCache[LSTM].from_model(LSTM(mconfig))
        train(project_path, tcache, dataset, config=tconfig)
    else:
        tcache = TrainCache[LSTM].load(project_path, LSTM)

    state_init = smd.sample_random_init_state()
    n_prop = 200

    # compute xs_real
    state_list = [state_init]
    for _ in range(n_prop):
        state_list.append(smd.propagate(state_list[-1]))
    xs_real = [s[0] for s in state_list]

    # compute xs_est and terminate flag est
    assert tcache.best_model is not None
    prop = Propagator(tcache.best_model, encoding_rule=rule)
    av = AngleVector(state_init)
    elem_dict = ElementDict([av])
    prop.feed(elem_dict)
    elem_dict_list = prop.predict(n_prop)
    xs_est = np.array([elem_dict[AngleVector].numpy()[0] for elem_dict in elem_dict_list])
    flags_est = np.array([elem_dict[TerminateFlag].numpy()[0] for elem_dict in elem_dict_list])

    plt.plot(xs_est, c="red", label="x_est")
    plt.plot(xs_real, c="blue", label="x_real")
    plt.plot(flags_est, c="green", label="prob. terminate")
    plt.legend()
    plt.show()
