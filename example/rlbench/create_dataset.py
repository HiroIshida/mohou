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
from rlbench.demo import Demo

from mohou.file import get_project_dir
from mohou.types import AngleVector, RGBImage, DepthImage
from mohou.types import ElementSequence, EpisodeData, MultiEpisodeChunk


def rlbench_demo_to_mohou_episode_data(demo: Demo) -> EpisodeData:
    seq_av = ElementSequence()  # type: ignore[var-annotated]
    seq_rgb = ElementSequence()  # type: ignore[var-annotated]
    seq_depth = ElementSequence()  # type: ignore[var-annotated]

    for obs in demo:
        av = AngleVector(np.array(obs.joint_positions.tolist() + [obs.gripper_open]))
        rgb = RGBImage(obs.overhead_rgb)
        depth = DepthImage(np.expand_dims(obs.overhead_depth, axis=2))

        rgb.resize((112, 112))
        depth.resize((112, 112))

        seq_av.append(av)
        seq_rgb.append(rgb)
        seq_depth.append(depth)
    return EpisodeData((seq_rgb, seq_depth, seq_av))


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

    mohou_episode_data_list = []
    for i in tqdm.tqdm(range(n_episode)):
        demo = task.get_demos(amount=1, live_demos=True)[0]
        mohou_episode_data = rlbench_demo_to_mohou_episode_data(demo)
        mohou_episode_data_list.append(mohou_episode_data)
    chunk = MultiEpisodeChunk(mohou_episode_data_list)
    chunk.dump(project_name)

    # create debug image
    filename = os.path.join(get_project_dir(project_name), "sample.gif")
    rgb_seq = chunk[0].filter_by_type(RGBImage)
    clip = ImageSequenceClip([img.numpy() for img in rgb_seq], fps=50)
    clip.write_gif(filename, fps=50)
