import argparse
import os
import tqdm

from moviepy.editor import ImageSequenceClip
import numpy as np
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointVelocity
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.environment import Environment
from rlbench.observation_config import ObservationConfig
from rlbench.tasks import CloseBox
from rlbench.backend.observation import Observation

from mohou.file import get_project_dir
from mohou.types import AngleVector, RGBDImage, RGBImage, DepthImage
from mohou.types import ElementSequence, EpisodeData, MultiEpisodeChunk


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-pn', type=str, default='rlbench_close_box', help='project name')
    parser.add_argument('-n', type=int, default=55, help='epoch num')
    args = parser.parse_args()
    n_episode = args.n
    project_name = args.pn

    # Data generation by rlbench
    obs_config = ObservationConfig()
    obs_config.set_all(True)

    env = Environment(
        action_mode=MoveArmThenGripper(
            arm_action_mode=JointVelocity(), gripper_action_mode=Discrete()),
        obs_config=ObservationConfig(),
        headless=True)
    env.launch()

    task = env.get_task(CloseBox)
    demos = []
    for i in tqdm.tqdm(range(n_episode)):
        demos.extend(task.get_demos(amount=1, live_demos=True))

    # data conversion from rlbench demos to mohou chunk
    data_list = []
    for demo in demos:
        seq_av = ElementSequence[AngleVector]()
        seq_rgb = ElementSequence[RGBImage]()
        seq_depth = ElementSequence[RGBDImage]()

        for obs in demo:
            av = AngleVector(obs.joint_positions)
            rgb = RGBImage(obs.overhead_rgb)
            depth = DepthImage(np.expand_dims(obs.overhead_depth, axis=2))

            rgb.resize((112, 112))
            depth.resize((112, 112))

            seq_av.append(av)
            seq_rgb.append(rgb)
            seq_depth.append(depth)

        data_list.append(EpisodeData((seq_rgb, seq_depth)))

    chunk = MultiEpisodeChunk(data_list)
    chunk.dump(project_name)

    # create debug image
    filename = os.path.join(get_project_dir(project_name), "sample.gif")
    rgb_seq = chunk[0].filter_by_type(RGBImage)
    clip = ImageSequenceClip([img.numpy() for img in rgb_seq], fps=50)
    clip.write_gif(filename, fps=50)
