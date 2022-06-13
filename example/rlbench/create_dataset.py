import argparse
import os
import uuid
from multiprocessing import Pool
from typing import Type

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
from rlbench.observation_config import ObservationConfig

from mohou.file import dump_object, get_subproject_path, load_objects
from mohou.types import (
    AngleVector,
    DepthImage,
    ElementSequence,
    EpisodeData,
    GripperState,
    MultiEpisodeChunk,
    RGBImage,
)


def rlbench_demo_to_mohou_episode_data(
    demo: Demo, camera_name: str, resolution: int
) -> EpisodeData:
    seq_av = ElementSequence()  # type: ignore[var-annotated]
    seq_gs = ElementSequence()  # type: ignore[var-annotated]
    seq_rgb = ElementSequence()  # type: ignore[var-annotated]
    seq_depth = ElementSequence()  # type: ignore[var-annotated]

    for obs in demo:
        av = AngleVector(obs.joint_positions)
        gs = GripperState(np.array([obs.gripper_open]))
        rgb = RGBImage(obs.__dict__[camera_name + "_rgb"])
        depth = DepthImage(np.expand_dims(obs.__dict__[camera_name + "_depth"], axis=2))

        rgb.resize((resolution, resolution))
        depth.resize((resolution, resolution))

        seq_av.append(av)
        seq_gs.append(gs)
        seq_rgb.append(rgb)
        seq_depth.append(depth)
    return EpisodeData.from_seq_list([seq_rgb, seq_depth, seq_av, seq_gs])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-pn", type=str, default="rlbench_close_box", help="project name")
    parser.add_argument("-tn", type=str, default="CloseDrawer", help="task name")
    parser.add_argument("-cn", type=str, default="overhead", help="camera name")
    parser.add_argument("-n", type=int, default=55, help="epoch num")
    parser.add_argument("-resol", type=int, default=112, help="epoch num")
    args = parser.parse_args()
    n_episode = args.n
    project_name = args.pn
    task_name = args.tn
    camera_name = args.cn
    resolution = args.resol

    assert camera_name in ["left_shoulder", "right_shoulder", "overhead", "wrist", "front"]
    assert resolution in [112, 224]

    def generate_demo(n_episode: int):
        # Data generation by rlbench
        obs_config = ObservationConfig()
        obs_config.set_all(True)

        env = Environment(
            action_mode=MoveArmThenGripper(
                arm_action_mode=JointVelocity(), gripper_action_mode=Discrete()
            ),
            obs_config=ObservationConfig(),
            headless=True,
        )
        env.launch()

        assert hasattr(rlbench.tasks, task_name)
        task_type: Type[Task] = getattr(rlbench.tasks, task_name)
        task = env.get_task(task_type)

        for i in tqdm.tqdm(range(n_episode)):
            demo = task.get_demos(amount=1, live_demos=True)[0]
            uuid_str = str(uuid.uuid4())
            dump_object(demo, project_name, str(uuid.uuid4()), subpath="temp")

    # First store demos in temporary files
    n_cpu = os.cpu_count()
    assert n_cpu is not None
    n_process = int(n_cpu * 0.5 - 1)
    n_process_list_assign = [len(lst) for lst in np.array_split(range(n_episode), n_process)]
    p = Pool(n_process)
    print("n_episode assigned to each process: {}".format(n_process_list_assign))
    p.map(generate_demo, n_process_list_assign)

    # load demos in temporary files and create chunks
    demos = load_objects(Demo, project_name, subpath="temp")
    episodes = [rlbench_demo_to_mohou_episode_data(demo, camera_name, resolution) for demo in demos]
    chunk = MultiEpisodeChunk.from_data_list(episodes)
    chunk.dump(project_name)

    # dump images
    gif_dir_path = get_subproject_path(project_name, "train_data_gif")
    for i, episode in enumerate(episodes):
        rgb_seq = episode.get_sequence_by_type(RGBImage)
        clip = ImageSequenceClip([img.numpy() for img in rgb_seq], fps=50)
        file_path = gif_dir_path / "sample{}.gif".format(i)
        clip.write_gif(str(file_path), fps=50)
