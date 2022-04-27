import argparse
import os

import tqdm
from moviepy.editor import ImageSequenceClip
import numpy as np
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointPosition
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.environment import Environment
from rlbench.observation_config import ObservationConfig
from rlbench.backend.observation import Observation
from rlbench.tasks import CloseDrawer

from mohou.file import get_project_dir
from mohou.types import RGBImage, DepthImage, AngleVector, ElementDict
from mohou.types import MultiEpisodeChunk
from mohou.default import create_default_propagator
from mohou.script_utils import auto_detect_autoencoder_type


def av_to_action(av: AngleVector) -> np.ndarray:
    return av.numpy()


def obs_to_elemdict(obs: Observation) -> ElementDict:
    av = AngleVector(np.array(obs.joint_positions.tolist() + [obs.gripper_open]))
    rgb = RGBImage(obs.overhead_rgb)
    rgb.resize((112, 112))
    depth = DepthImage(np.expand_dims(obs.overhead_depth, axis=2))
    depth.resize((112, 112))
    return ElementDict([av, rgb, depth])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-pn', type=str, default='rlbench_close_box', help='project name')
    parser.add_argument('-n', type=int, default=250, help='step num')
    args = parser.parse_args()
    project_name = args.pn
    n_step = args.n

    obs_config = ObservationConfig()
    obs_config.set_all(True)

    env = Environment(
        action_mode=MoveArmThenGripper(
            arm_action_mode=JointPosition(), gripper_action_mode=Discrete()),
        obs_config=ObservationConfig(),
        headless=True)
    env.launch()

    task = env.get_task(CloseDrawer)
    task.reset()

    chunk = MultiEpisodeChunk.load(project_name)
    av_init = chunk.data_list_intact[0].filter_by_primitive_type(AngleVector)[0]

    ae_type = auto_detect_autoencoder_type(project_name)
    prop = create_default_propagator(project_name, n_angle_vector=7 + 1, ae_type=ae_type)  # 1 for gripper

    rgb_seq_gif = []

    obs, _, _ = task.step(av_to_action(av_init))
    edict = obs_to_elemdict(obs)
    prop.feed(edict)
    for i in tqdm.tqdm(range(n_step)):
        edict_next = prop.predict(n_prop=1)[0]
        av_next = edict_next[AngleVector]
        obs, _, _ = task.step(av_to_action(av_next))
        edict = obs_to_elemdict(obs)
        prop.feed(edict)

        rgb_seq_gif.append(RGBImage(obs.overhead_rgb))

    filename = os.path.join(get_project_dir(project_name), "feedback_simulation.gif")
    clip = ImageSequenceClip([img.numpy() for img in rgb_seq_gif], fps=50)
    clip.write_gif(filename, fps=50)
