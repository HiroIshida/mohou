import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pybullet as pb
import pybullet_data
import tqdm
from moviepy.editor import ImageSequenceClip
from pybullet_utils import BoxConfig, create_box
from skrobot.coordinates import Coordinates
from skrobot.coordinates.geo import orient_coords_to_axis
from skrobot.coordinates.math import (
    quaternion2matrix,
    rotation_matrix_from_axis,
    rpy2quaternion,
    wxyz2xyzw,
    xyzw2wxyz,
)
from skrobot.model import RobotModel
from skrobot.models.urdf import RobotModelFromURDF

from mohou.file import create_project_dir, get_project_path
from mohou.types import (
    AngleVector,
    ElementDict,
    EpisodeBundle,
    EpisodeData,
    GripperState,
    MetaData,
    RGBImage,
)


class IKFailError(Exception):
    pass


@dataclass
class Camera:
    """Camera
    Most of the functions in this class
    are took from https://github.com/kosuke55/hanging_points_cnn
    Copyright (c)  2021 Kosuke Takeuchi
    """

    coords: Coordinates
    resolution: int

    def draw_camera_pos(self):
        pb.removeAllUserDebugItems()
        start = self.coords.worldpos()
        end_x = start + self.coords.rotate_vector([0.1, 0, 0])
        pb.addUserDebugLine(start, end_x, [1, 0, 0], 3)
        end_y = start + self.coords.rotate_vector([0, 0.1, 0])
        pb.addUserDebugLine(start, end_y, [0, 1, 0], 3)
        end_z = start + self.coords.rotate_vector([0, 0, 0.1])
        pb.addUserDebugLine(start, end_z, [0, 0, 1], 3)

    def look_at(self, p: np.ndarray, horizontal=False):
        if np.all(p == self.coords.worldpos()):
            return
        z = p - self.coords.worldpos()
        orient_coords_to_axis(self.coords, z)
        if horizontal:
            self.coords.newcoords(
                Coordinates(
                    pos=self.coords.worldpos(),
                    rot=rotation_matrix_from_axis(z, [0, 0, -1], axes="zy"),
                )
            )

    def render(self):
        target = self.coords.worldpos() + self.coords.rotate_vector([0, 0, 1.0])
        up = self.coords.rotate_vector([0, -1.0, 0])
        vm = pb.computeViewMatrix(self.coords.worldpos(), target, up)
        fov, aspect, near, far = 45.0, 1.0, 0.01, 5.1
        pm = pb.computeProjectionMatrixFOV(fov, aspect, near, far)
        _, _, rgba, depth, _ = pb.getCameraImage(
            self.resolution,
            self.resolution,
            vm,
            pm,
            renderer=pb.ER_TINY_RENDERER,
        )
        rgb = rgba[:, :, :3]
        return rgb


@dataclass
class Environment:
    handles: Dict[str, int]
    joint_to_id_table: Dict[str, int]
    link_to_id_table: Dict[str, int]
    urdf_path: str
    latest_command: Dict[str, float]
    elapsed_time: int = 0
    default_callback: Optional[Callable] = None

    @classmethod
    def create(cls) -> "Environment":
        pb_data_path = Path(pybullet_data.__file__).parent
        panda_urdf_path = pb_data_path / "franka_panda/panda.urdf"

        pb.setPhysicsEngineParameter(numSolverIterations=100)
        pb.setTimeStep(timeStep=0.02)
        pb.setAdditionalSearchPath(pybullet_data.getDataPath())  # used by loadURDF
        pb.loadURDF("plane.urdf")
        robot_id = pb.loadURDF(str(panda_urdf_path), useFixedBase=True)
        pb.configureDebugVisualizer(pb.COV_ENABLE_GUI, 0)
        pb.setGravity(0, 0, -10)

        box_id = create_box(BoxConfig(size=(0.3, 0.2, 0.08), rgba=(1, 0.7, 0.7, 1.0)), friction=3.0)
        box2_id = create_box(
            BoxConfig(size=(0.3, 0.2, 0.04), rgba=(0.7, 1.0, 0.7, 1.0)), friction=0.5
        )

        handles = {"robot": robot_id, "box1": box_id, "box2": box2_id}

        joint_table = {}
        link_table = {pb.getBodyInfo(robot_id)[0].decode("UTF-8"): -1}
        for idx in range(pb.getNumJoints(robot_id)):
            joint_info = pb.getJointInfo(robot_id, idx)
            joint_id = joint_info[0]
            joint_name = joint_info[1].decode("UTF-8")
            joint_table[joint_name] = joint_id

            tmp = joint_info[12].decode("UTF-8")
            name = "_".join(tmp.split("/"))
            link_table[name] = idx

        return cls(handles, joint_table, link_table, str(panda_urdf_path), {})

    def __post_init__(self):
        self.reset_world()

    def reset_world(self, x_bias: float = 0.0, y_bias: float = 0.0, yaw_bias: float = 0.0):
        self.elapsed_time = 0

        # initialize latest command
        joint_names = list(self.joint_to_id_table.keys())
        angles = self.get_angle_vector(joint_names)
        for name, angle in zip(joint_names, angles):
            self.latest_command[name] = angle

        # reset robot
        robot_id = self.handles["robot"]
        for joint_id in self.joint_to_id_table.values():
            pb.resetJointState(robot_id, joint_id, 0.0, targetVelocity=0.0)

        # reset base
        for object_id in self.handles.values():
            pb.resetBaseVelocity(
                object_id, linearVelocity=(0.0, 0.0, 0.0), angularVelocity=(0.0, 0.0, 0.0)
            )

        x_pos = 0.45 + x_bias
        y_pos = 0.1 + y_bias
        yaw = 0.0 + yaw_bias
        self.set_pose("box1", (x_pos, y_pos, 0.05), (yaw, 0, 0))
        self.set_pose("box2", (x_pos, y_pos, 0.08), (yaw, 0, 0))

    def set_pose(
        self,
        body_name: str,
        point: Tuple[float, float, float],
        rpy: Tuple[float, float, float] = (0, 0, 0),
    ):
        body_id = self.handles[body_name]
        q = rpy2quaternion(rpy)
        pb.resetBasePositionAndOrientation(body_id, point, wxyz2xyzw(q))

    def get_skrobot_coords(self, body_name: str, link_name: Optional[str] = None) -> Coordinates:
        # NOTE quat is xyzw order
        body_id = self.handles[body_name]
        if link_name is None:
            trans, quat = pb.getBasePositionAndOrientation(body_id)
        else:
            link_id = self.link_to_id_table[link_name]
            ret = pb.getLinkState(body_id, link_id, computeForwardKinematics=1)
            trans, quat = ret[0], ret[1]
        mat = quaternion2matrix(xyzw2wxyz(quat))
        return Coordinates(trans, mat)

    def set_joint_angle(self, joint_name: str, angle: float, gain: float = 1.0):
        self.latest_command[joint_name] = angle

        joint_id = self.joint_to_id_table[joint_name]
        pb.setJointMotorControl2(
            bodyIndex=self.handles["robot"],
            jointIndex=joint_id,
            controlMode=pb.POSITION_CONTROL,
            targetPosition=angle,
            targetVelocity=0.0,
            force=300,
            positionGain=gain,
            velocityGain=1.0,
            maxVelocity=1.0,
        )

    def set_angle_vetor(self, robot: "PandaModel"):
        for name in robot.control_joint_names:
            joint = robot.robot_model.__dict__[name]
            joint_id = self.joint_to_id_table[name]
            pb.resetJointState(self.handles["robot"], joint_id, joint.joint_angle())

    def send_angle_vector(self, robot: "PandaModel"):
        for name in robot.control_joint_names:
            joint = robot.robot_model.__dict__[name]
            self.set_joint_angle(name, joint.joint_angle())

    def get_angle_vector(self, joint_names: List[str]) -> np.ndarray:
        angle_list = []
        for name in joint_names:
            joint_id = self.joint_to_id_table[name]
            pos, _, _, _ = pb.getJointState(self.handles["robot"], joint_id)
            angle_list.append(pos)
        return np.array(angle_list)

    def change_gripper_position(self, pos: float) -> None:
        assert pos > -1e-3 and pos < 0.08 + 1e-3
        name1 = "panda_finger_joint1"
        name2 = "panda_finger_joint2"
        self.set_joint_angle(name1, pos * 0.5, True)
        self.set_joint_angle(name2, pos * 0.5, True)

    def get_gripper_position_target(self) -> float:
        name1 = "panda_finger_joint1"
        name2 = "panda_finger_joint2"
        pos1 = self.latest_command[name1]
        pos2 = self.latest_command[name2]
        return pos1 + pos2

    def wait_interpolation(self, sleep: float = 0.0, callback: Optional[Callable] = None) -> None:
        while True:
            self.step(1, sleep, callback=callback)
            velocities = []
            for joint_id in self.joint_to_id_table.values():
                _, vel, _, _ = pb.getJointState(self.handles["robot"], joint_id)
                velocities.append(vel)
            vel_max = np.max(np.abs(velocities))
            if vel_max < 0.05:
                break

    def step(self, n: int, sleep: float = 0.0, callback: Optional[Callable] = None) -> None:
        for _ in range(n):
            pb.stepSimulation()

            if callback is not None:
                callback(self)
            else:
                if self.default_callback is not None:
                    self.default_callback(self)

            self.elapsed_time += 1
            time.sleep(sleep)


@dataclass
class PandaModel:
    robot_model: RobotModel

    @classmethod
    def from_urdf(cls, urdf_path: str) -> "PandaModel":
        robot_model = RobotModelFromURDF(urdf_file=urdf_path)
        return cls(robot_model)

    def solve_ik(self, coords: Coordinates) -> None:
        joints = [self.robot_model.__dict__[jname] for jname in self.control_joint_names]
        link_list = [joint.child_link for joint in joints]
        end_effector = self.robot_model.__dict__[self.end_effector_name]
        av_next = self.robot_model.inverse_kinematics(coords, end_effector, link_list)
        solved = isinstance(av_next, np.ndarray)
        if not solved:
            raise IKFailError

    def set_angle_vector(self, angles: List[float]):
        for name, angle in zip(self.control_joint_names, angles):
            joint = self.robot_model.__dict__[name]
            joint.joint_angle(angle)

    @property
    def control_joint_names(self) -> List[str]:
        return ["panda_joint{}".format(i + 1) for i in range(7)]

    @property
    def end_effector_name(self) -> str:
        return "panda_grasptarget"

    def move_end_pos(self, pos, wrt: str = "local") -> None:
        pos = np.array(pos, dtype=np.float64)
        co_end_link = self.robot_model.__dict__[self.end_effector_name].copy_worldcoords()
        co_end_link.translate(pos, wrt=wrt)
        self.solve_ik(co_end_link)

    def move_end_rot(self, angle, axis, wrt: str = "local") -> None:
        co_end_link = self.robot_model.__dict__[self.end_effector_name].copy_worldcoords()
        co_end_link.rotate(angle, axis, wrt=wrt)
        self.solve_ik(co_end_link)


class Task:
    camera: Camera
    env: Environment
    robot: PandaModel

    def __init__(self, camera: Camera):
        self.camera = camera
        self.env = Environment.create()
        self.robot = PandaModel.from_urdf(self.env.urdf_path)

    def reset(self, biases) -> None:
        self.env.reset_world(*biases)
        self.robot.set_angle_vector([0.0, 0.7, 0.0, -0.5, 0.0, 1.3, -0.8])
        self.env.set_angle_vetor(self.robot)
        self.env.change_gripper_position(0.07)

    def is_successful(self) -> bool:
        self.env.step(100, 0)
        co = self.env.get_skrobot_coords("box2")
        return co.translation[2] > 0.2

    def replay(self, episode: EpisodeData, global_sleep=0.0) -> None:
        width: int = episode.metadata["sampling_width"]  # type: ignore
        av_seq = episode.get_sequence_by_type(AngleVector)
        gs_seq = episode.get_sequence_by_type(GripperState)
        for av, gs in zip(av_seq, gs_seq):
            self.robot.set_angle_vector(list(av.numpy()))
            self.env.send_angle_vector(self.robot)
            self.env.change_gripper_position(gs.numpy().item())
            self.env.step(width, sleep=global_sleep)

    def run_prescribed_motion(self, global_sleep=0.0) -> EpisodeData:
        edict_list: List[ElementDict] = []

        def callback(env: Environment):
            av = env.get_angle_vector(self.robot.control_joint_names)
            gs = env.get_gripper_position_target()
            rgb = self.camera.render()
            edict = ElementDict([AngleVector(av), GripperState(np.array([gs])), RGBImage(rgb)])
            edict_list.append(edict)

        self.env.default_callback = callback

        # pre-push pose
        target = self.env.get_skrobot_coords("box2").copy_worldcoords()
        target.translate([0.0, 0.15, 0.04])
        target.rotate(np.pi * 0.5, "y")
        target.rotate(np.pi * 0.5, "x")

        self.robot.solve_ik(target)
        self.env.send_angle_vector(self.robot)
        self.env.wait_interpolation(sleep=global_sleep)

        # push
        target.translate([0.0, -0.12, 0.0], wrt="world")
        self.robot.solve_ik(target)
        self.env.send_angle_vector(self.robot)
        self.env.wait_interpolation(sleep=global_sleep)

        # go up
        target.translate([0.0, 0.0, 0.1], wrt="world")
        self.robot.solve_ik(target)
        self.env.send_angle_vector(self.robot)
        self.env.wait_interpolation(sleep=global_sleep)

        # move around
        target = self.env.get_skrobot_coords("box2").copy_worldcoords()
        target.translate([0.0, -0.25, 0.12])
        target.rotate(np.pi * 0.5, "y")
        self.robot.solve_ik(target)
        self.env.send_angle_vector(self.robot)
        self.env.wait_interpolation(sleep=global_sleep)

        target = self.env.get_skrobot_coords("box2").copy_worldcoords()
        target.translate([0.0, -0.23, 0.04])
        target.rotate(np.pi * 0.5, "y")
        target.rotate(np.pi * 0.1, "z")
        self.robot.solve_ik(target)
        self.env.send_angle_vector(self.robot)
        self.env.wait_interpolation(sleep=global_sleep)

        target = self.env.get_skrobot_coords("box2").copy_worldcoords()
        target.translate([0.0, -0.18, 0.0])
        target.rotate(np.pi * 0.5, "y")
        target.rotate(np.pi * 0.43, "z")
        self.robot.solve_ik(target)
        self.env.send_angle_vector(self.robot)
        self.env.wait_interpolation(sleep=global_sleep)

        # open and grasp
        self.env.change_gripper_position(0.07)
        self.env.wait_interpolation()
        self.robot.move_end_pos([0.11, 0.0, 0])
        self.env.send_angle_vector(self.robot)
        self.env.step(20, sleep=global_sleep)
        self.env.change_gripper_position(0.02)
        self.env.step(20, sleep=global_sleep)

        # lift
        self.robot.move_end_pos(pos=(0, 0, 0.15), wrt="world")
        self.robot.move_end_rot(np.pi * 0.15, "z")
        self.env.send_angle_vector(self.robot)
        self.env.wait_interpolation(sleep=global_sleep)

        self.env.default_callback = None
        sampling_width = 2
        metadata = MetaData(sampling_width=sampling_width)
        episode = EpisodeData.from_edict_list(edict_list[::sampling_width], metadata=metadata)
        return episode


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--feedback", action="store_true", help="feedback mode")
    parser.add_argument("-pn", type=str, default="panda_pickup_book", help="project name")
    parser.add_argument("-pp", type=str, help="project path name. preferred over pn.")
    parser.add_argument("-n", type=int, default=105, help="epoch num")
    parser.add_argument("-m", type=int, default=224, help="pixel num")
    parser.add_argument("-untouch", type=int, default=5, help="num of untouch episode")
    parser.add_argument("-seed", type=int, default=1, help="seed")

    args = parser.parse_args()
    n_epoch: int = args.n
    n_pixel: int = args.m
    feedback_mode: bool = args.feedback
    project_name: str = args.pn
    n_untouch: int = args.untouch
    seed: int = args.seed
    project_path_str: Optional[str] = args.pp

    assert n_epoch - n_untouch > 0

    if project_path_str is None:
        assert project_name is not None
        create_project_dir(project_name)
        project_path = get_project_path(project_name)
    else:
        project_path = Path(project_path_str)
        project_path.mkdir(exist_ok=True)

    np.random.seed(seed)

    # create task
    pb.connect(pb.DIRECT)
    camera = Camera(Coordinates((1.9, 0, 0.7)), n_pixel)
    camera.look_at(np.array([0.5, 0, 0.3]), horizontal=True)
    task = Task(camera)

    # create bundle
    episode_list: List[EpisodeData] = []

    with tqdm.tqdm(total=n_epoch) as pbar:
        while len(episode_list) < n_epoch:
            x_bias = np.random.randn() * 0.06
            y_bias = np.random.randn() * 0.06
            yaw_bias = np.random.randn() * 0.1

            task.reset((x_bias, y_bias, yaw_bias))
            try:
                episode = task.run_prescribed_motion()
            except IKFailError:
                continue
            is_rollout_successful = task.is_successful()

            task.reset((x_bias, y_bias, yaw_bias))
            task.replay(episode)
            is_replay_successful = task.is_successful()

            if is_rollout_successful and is_replay_successful:
                episode_list.append(episode)
                pbar.update(1)

    bundle = EpisodeBundle.from_episodes(episode_list)
    bundle.dump(project_path)
    bundle.plot_vector_histories(AngleVector, project_path)

    # dump debug gif
    img_seq = bundle[0].get_sequence_by_type(RGBImage)
    file_path = project_path / "sample.gif"
    clip = ImageSequenceClip([img for img in img_seq], fps=50)
    clip.write_gif(str(file_path), fps=50)
