import argparse
import os
import shutil
import uuid
from multiprocessing import Pool
from pathlib import Path
from typing import List, Type

import numpy as np
import rlbench.tasks
import tqdm
from moviepy.editor import ImageSequenceClip
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointVelocity
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.backend.task import Task
from rlbench.demo import Demo
from rlbench.environment import Environment
from utils import setup_observation_config

from mohou.file import dump_object, get_project_path, get_subproject_path, load_objects
from mohou.types import (
    AngleVector,
    DepthImage,
    ElementSequence,
    EpisodeBundle,
    EpisodeData,
    GripperState,
    RGBImage,
)


def rlbench_demo_to_mohou_episode_data(
    demo: Demo, camera_name: str, resolution: int
) -> EpisodeData:
    av_list: List[AngleVector] = []
    gs_list: List[GripperState] = []
    rgb_list: List[RGBImage] = []
    depth_list: List[DepthImage] = []

    for obs in demo:
        av = AngleVector(obs.joint_positions)
        gs = GripperState(np.array([obs.gripper_open]))
        rgb = RGBImage(obs.__dict__[camera_name + "_rgb"])
        depth = DepthImage(np.expand_dims(obs.__dict__[camera_name + "_depth"], axis=2))

        rgb.resize((resolution, resolution))
        depth.resize((resolution, resolution))

        av_list.append(av)
        gs_list.append(gs)
        rgb_list.append(rgb)
        depth_list.append(depth)
    return EpisodeData.from_seq_list(
        [
            ElementSequence(av_list),
            ElementSequence(gs_list),
            ElementSequence(rgb_list),
            ElementSequence(depth_list),
        ]
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-pn", type=str, default="rlbench_close_box", help="project name")
    parser.add_argument("-tn", type=str, default="CloseDrawer", help="task name")
    parser.add_argument("-cn", type=str, default="overhead", help="camera name")
    parser.add_argument("-n", type=int, default=55, help="epoch num")
    parser.add_argument("-p", type=int, default=0, help="number of processes")
    parser.add_argument("-resol", type=int, default=112, help="epoch num")
    args = parser.parse_args()
    n_episode: int = args.n
    project_name: str = args.pn
    task_name: str = args.tn
    camera_name: str = args.cn
    resolution: int = args.resol
    n_process: int = args.p
    assert n_process > -1

    camera_names = {"left_shoulder", "right_shoulder", "overhead", "wrist", "front"}

    assert camera_name in camera_names
    assert resolution in [112, 224]

    def generate_demo(n_episode: int):
        # Data generation by rlbench
        env = Environment(
            action_mode=MoveArmThenGripper(
                arm_action_mode=JointVelocity(), gripper_action_mode=Discrete()
            ),
            obs_config=setup_observation_config(camera_name, resolution),
            headless=True,
        )
        env.launch()

        assert hasattr(rlbench.tasks, task_name)
        task_type: Type[Task] = getattr(rlbench.tasks, task_name)
        task = env.get_task(task_type)

        for i in tqdm.tqdm(range(n_episode)):
            demo = task.get_demos(amount=1, live_demos=True)[0]
            dump_object(demo, project_name, str(uuid.uuid4()), subpath=Path("temp"))

    # delete temp files
    temp_path = get_subproject_path(project_name, subpath=Path("temp"))
    shutil.rmtree(str(temp_path))

    # First store demos in temp files
    # Unlike python's tempfile, we do not clear these temp files after this function for the easy of debugging
    if n_process == 0:
        n_cpu = os.cpu_count()
        assert n_cpu is not None
        n_process = int(n_cpu * 0.5 - 1)
    n_process_list_assign = [len(lst) for lst in np.array_split(range(n_episode), n_process)]

    p = Pool(n_process)
    print("n_episode assigned to each process: {}".format(n_process_list_assign))
    p.map(generate_demo, n_process_list_assign)

    # load demos in temporary files and create bundles
    demos = load_objects(Demo, project_name, subpath=Path("temp"))
    episodes = [rlbench_demo_to_mohou_episode_data(demo, camera_name, resolution) for demo in demos]
    bundle = EpisodeBundle.from_data_list(episodes)
    bundle.dump(get_project_path(project_name))

    # dump images
    gif_dir_path = get_subproject_path(project_name, "train_data_gif")
    for i, episode in enumerate(episodes):
        rgb_seq = episode.get_sequence_by_type(RGBImage)
        clip = ImageSequenceClip([img.numpy() for img in rgb_seq], fps=50)
        file_path = gif_dir_path / "sample{}.gif".format(i)
        clip.write_gif(str(file_path), fps=50)
