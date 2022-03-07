import argparse
import os
import uuid
import pickle
import multiprocessing
import tqdm
import tempfile

from moviepy.editor import ImageSequenceClip
import numpy as np
import psutil
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
    parser.add_argument('-m', type=int, default=-1, help='multi process num')
    args = parser.parse_args()
    n_episode = args.n
    n_process = args.m
    project_name = args.pn

    with tempfile.TemporaryDirectory() as td:

        def create_demos(arg):
            cpu_idx, n_data_gen = arg
            disable_tqdm = (cpu_idx != 0)

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
            for _ in tqdm.tqdm(range(n_data_gen), disable=disable_tqdm):
                demo = task.get_demos(amount=1, live_demos=True)[0]

                with open(os.path.join(td, str(uuid.uuid4()) + '.pkl'), 'wb') as f:
                    pickle.dump(demo, f)

        if n_process == -1:
            n_cpu = psutil.cpu_count(logical=False)
            print('{} physical cpus are detected'.format(n_cpu))
            n_process = n_cpu

        pool = multiprocessing.Pool(n_process)
        n_assigh_list = [len(lst) for lst in np.array_split(range(n_episode), n_process)]
        pool.map(create_demos, zip(range(n_process), n_assigh_list))

        demos = []
        for filename in os.listdir(td):
            with open(os.path.join(td, filename), 'rb') as f:
                demos.append(pickle.load(f))

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
