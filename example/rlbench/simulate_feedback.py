import argparse
from typing import Type

import numpy as np
import rlbench.tasks
import tqdm
from moviepy.editor import ImageSequenceClip
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointPosition
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.backend.observation import Observation
from rlbench.backend.task import Task
from rlbench.environment import Environment
from utils import setup_observation_config

from mohou.default import create_default_propagator
from mohou.file import get_project_path
from mohou.types import (
    AngleVector,
    DepthImage,
    ElementDict,
    EpisodeBundle,
    GripperState,
    RGBImage,
)


def edict_to_action(edict: ElementDict) -> np.ndarray:
    av_next = edict[AngleVector]
    gs_next = edict[GripperState]
    return np.hstack([av_next.numpy(), gs_next.numpy()])


def obs_to_edict(obs: Observation, resolution: int, camera_name: str) -> ElementDict:
    av = AngleVector(obs.joint_positions)
    gs = GripperState(np.array([obs.gripper_open]))

    arr_rgb = obs.__dict__[camera_name + "_rgb"]
    arr_depth = np.expand_dims(obs.__dict__[camera_name + "_depth"], axis=2)

    rgb = RGBImage(arr_rgb)
    depth = DepthImage(arr_depth)
    return ElementDict([av, gs, rgb, depth])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-pn", type=str, default="rlbench_close_box", help="project name")
    parser.add_argument("-tn", type=str, default="CloseDrawer", help="task name")
    parser.add_argument("-cn", type=str, default="overhead", help="camera name")
    parser.add_argument("-n", type=int, default=250, help="step num")
    parser.add_argument("-m", type=int, default=3, help="simulation num")
    args = parser.parse_args()
    project_name: str = args.pn
    task_name: str = args.tn
    camera_name: str = args.cn
    n_step: int = args.n
    n_sim: int = args.m

    project_path = get_project_path(project_name)

    bundle = EpisodeBundle.load(project_path)
    resolution = bundle.spec.type_shape_table[RGBImage][0]

    obs_config = setup_observation_config(camera_name, resolution)
    env = Environment(
        action_mode=MoveArmThenGripper(
            arm_action_mode=JointPosition(), gripper_action_mode=Discrete()
        ),
        obs_config=obs_config,
        headless=True,
    )
    env.launch()

    untouch_bundle = bundle.get_untouch_bundle()
    av_init = untouch_bundle[0].get_sequence_by_type(AngleVector)[0]
    gs_init = untouch_bundle[0].get_sequence_by_type(GripperState)[0]
    edict_init = ElementDict([av_init, gs_init])

    assert hasattr(rlbench.tasks, task_name)
    task_type: Type[Task] = getattr(rlbench.tasks, task_name)
    task = env.get_task(task_type)

    for i in range(n_sim):
        task.reset()

        prop = create_default_propagator(project_path)

        rgb_seq_gif = []

        obs, _, _ = task.step(edict_to_action(edict_init))
        edict = obs_to_edict(obs, resolution, camera_name)
        prop.feed(edict)
        for _ in tqdm.tqdm(range(n_step)):
            edict_next = prop.predict(n_prop=1)[0]
            obs, _, _ = task.step(edict_to_action(edict_next))
            edict = obs_to_edict(obs, resolution, camera_name)
            prop.feed(edict)

            rgb_seq_gif.append(RGBImage(obs.__dict__[camera_name + "_rgb"]))

        file_path = project_path / "feedback_simulation-{}.gif".format(i)
        clip = ImageSequenceClip([img.numpy() for img in rgb_seq_gif], fps=50)
        clip.write_gif(str(file_path), fps=50)
