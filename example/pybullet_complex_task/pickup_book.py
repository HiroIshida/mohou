import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pybullet as pb
import pybullet_data
from skrobot.coordinates import Coordinates
from skrobot.coordinates.math import (
    quaternion2matrix,
    rpy2quaternion,
    wxyz2xyzw,
    xyzw2wxyz,
)
from skrobot.model import RobotModel
from skrobot.models.urdf import RobotModelFromURDF
from utils import BoxConfig, create_box


class IKFailError(Exception):
    pass


@dataclass
class Environment:
    handles: Dict[str, int]
    joint_to_id_table: Dict[str, int]
    link_to_id_table: Dict[str, int]
    urdf_path: str

    def __post_init__(self):
        # reset robot
        robot_id = self.handles["robot"]
        for joint_id in self.joint_to_id_table.values():
            pb.resetJointState(robot_id, joint_id, 0.0, targetVelocity=0.0)

        # reset base
        for object_id in self.handles.values():
            pb.resetBaseVelocity(
                object_id, linearVelocity=(0.0, 0.0, 0.0), angularVelocity=(0.0, 0.0, 0.0)
            )

        self.randomize_world()

    def randomize_world(self):
        x_pos = 0.45 + np.random.randn() * 0.05
        y_pos = 0.1 + np.random.randn() * 0.05
        yaw = np.random.randn() * 0.1
        np.random.randn(3) * 0.06
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

    @classmethod
    def create(cls) -> "Environment":
        pb_data_path = Path(pybullet_data.__file__).parent
        panda_urdf_path = pb_data_path / "franka_panda/panda.urdf"

        pb.connect(pb.GUI)
        pb.setPhysicsEngineParameter(numSolverIterations=50)
        pb.setTimeStep(timeStep=0.001)
        pb.setAdditionalSearchPath(pybullet_data.getDataPath())  # used by loadURDF
        pb.loadURDF("plane.urdf")
        robot_id = pb.loadURDF(str(panda_urdf_path), useFixedBase=True)
        pb.configureDebugVisualizer(pb.COV_ENABLE_GUI, 0)
        pb.setGravity(0, 0, -10)

        box_id = create_box(BoxConfig(size=(0.3, 0.2, 0.08), rgba=(1, 0.7, 0.7, 1.0)))
        box2_id = create_box(BoxConfig(size=(0.3, 0.2, 0.04), rgba=(0.7, 1.0, 0.7, 1.0)))

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

        return cls(handles, joint_table, link_table, str(panda_urdf_path))

    def set_joint_angle(self, joint_name: str, angle: float, gain: float = 1.0):
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

    def send_angel_vector(self, robot: "PandaModel"):
        for name in robot.control_joint_names:
            joint = robot.robot_model.__dict__[name]
            self.set_joint_angle(name, joint.joint_angle())

    def change_gripper_position(self, pos: float):
        assert pos > -1e-3 and pos < 0.08 + 1e-3
        name1 = "panda_finger_joint1"
        name2 = "panda_finger_joint2"
        self.set_joint_angle(name1, pos * 0.5, True)
        self.set_joint_angle(name2, pos * 0.5, True)

    def wait_interpolation(self, sleep: float = 0.0, callback: Optional[Callable] = None) -> None:
        counter = 0
        while True:
            time.sleep(sleep)
            pb.stepSimulation()
            if callback is not None:
                callback()
            velocities = []
            for joint_id in self.joint_to_id_table.values():
                _, vel, _, _ = pb.getJointState(self.handles["robot"], joint_id)
                velocities.append(vel)
            vel_max = np.max(np.abs(velocities))
            counter += 1
            if vel_max < 0.05:
                break
        print(counter)

    def step(self, n: int, sleep: float = 0.0) -> None:
        for _ in range(n):
            pb.stepSimulation()
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


env = Environment.create()
robot = PandaModel.from_urdf(env.urdf_path)

for _ in range(30):
    env.randomize_world()
    robot.set_angle_vector([0.0, 0.7, 0.0, -0.5, 0.0, 1.3, -0.8])
    env.set_angle_vetor(robot)

    # pre-push pose
    target = env.get_skrobot_coords("box2").copy_worldcoords()
    target.translate([0.0, 0.15, 0.04])
    target.rotate(np.pi * 0.5, "y")

    robot.solve_ik(target)
    env.send_angel_vector(robot)
    env.wait_interpolation()
    # time.sleep(2)

    # push
    target.translate([0.0, -0.06, 0.0], wrt="world")
    robot.solve_ik(target)
    env.send_angel_vector(robot)
    env.wait_interpolation()

    # go up
    target.translate([0.0, 0.0, 0.1], wrt="world")
    robot.solve_ik(target)
    env.send_angel_vector(robot)
    env.wait_interpolation()

    # move around
    target = env.get_skrobot_coords("box2").copy_worldcoords()
    target.translate([0.0, -0.25, 0.12])
    target.rotate(np.pi * 0.5, "y")
    robot.solve_ik(target)
    env.send_angel_vector(robot)
    env.wait_interpolation()

    target = env.get_skrobot_coords("box2").copy_worldcoords()
    target.translate([0.0, -0.18, 0.04])
    target.rotate(np.pi * 0.5, "y")
    target.rotate(np.pi * 0.1, "z")
    robot.solve_ik(target)
    env.send_angel_vector(robot)
    env.wait_interpolation()

    target = env.get_skrobot_coords("box2").copy_worldcoords()
    target.translate([0.0, -0.18, 0.04])
    target.rotate(np.pi * 0.5, "y")
    target.rotate(np.pi * 0.35, "z")
    robot.solve_ik(target)
    env.send_angel_vector(robot)
    env.wait_interpolation()

    # open and grasp
    env.change_gripper_position(0.07)
    env.wait_interpolation()
    robot.move_end_pos([0.09, 0.0, 0])
    env.send_angel_vector(robot)
    env.step(300, sleep=0.0)
    env.change_gripper_position(0.02)
    env.step(300, sleep=0.0)
    # env.wait_interpolation(sleep=0.01)

    # lift
    robot.move_end_pos(pos=(0, 0, 0.15), wrt="world")
    robot.move_end_rot(np.pi * 0.15, "z")
    env.send_angel_vector(robot)
    env.wait_interpolation()
