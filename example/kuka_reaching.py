import argparse
import multiprocessing
import os
import pickle
import tempfile
import uuid
from pathlib import Path
from typing import List, Optional

import numpy as np
import psutil
import pybullet as pb
import pybullet_data
import tinyfk
import tqdm
from moviepy.editor import ImageSequenceClip

from mohou.default import create_default_propagator, load_default_image_encoder
from mohou.file import create_project_dir, get_project_path
from mohou.propagator import Propagator
from mohou.types import (
    AngleVector,
    DepthImage,
    ElementDict,
    ElementSequence,
    EpisodeBundle,
    EpisodeData,
    RGBImage,
)


class BulletManager(object):
    def __init__(self, use_gui, urdf_path, end_effector_name):
        client = pb.connect(pb.GUI if use_gui else pb.DIRECT)
        pb.configureDebugVisualizer(pb.COV_ENABLE_GUI, 0)
        robot_id = pb.loadURDF(urdf_path)
        pbdata_path = pybullet_data.getDataPath()
        pb.loadURDF(os.path.join(pbdata_path, "samurai.urdf"))

        link_table = {pb.getBodyInfo(robot_id, physicsClientId=client)[0].decode("UTF-8"): -1}
        joint_table = {}

        def heck(path):
            return "_".join(path.split("/"))

        for _id in range(pb.getNumJoints(robot_id, physicsClientId=client)):
            joint_info = pb.getJointInfo(robot_id, _id, physicsClientId=client)
            joint_id = joint_info[0]
            joint_name = joint_info[1].decode("UTF-8")
            joint_table[joint_name] = joint_id
            name_ = joint_info[12].decode("UTF-8")
            name = heck(name_)
            link_table[name] = _id

        self._box_id = None

        self._client = client
        self._robot_id = robot_id
        self._link_table = link_table
        self._joint_table = joint_table

        self._kin_solver = tinyfk.RobotModel(urdf_path)
        self._tinyfk_joint_ids = self._kin_solver.get_joint_ids(self.joint_names)
        self._tinyfk_endeffector_id = self._kin_solver.get_link_ids([end_effector_name])[0]

    @property
    def joint_names(self):
        return list(self._joint_table.keys())

    @property
    def joint_ids(self):
        return list(self._joint_table.values())

    def joint_angles(self):
        return np.array(
            [
                pb.getJointState(self._robot_id, joint_id, physicsClientId=self._client)[0]
                for joint_id in self.joint_ids
            ]
        )

    def set_joint_angles(self, joint_angles):
        assert len(joint_angles) == len(self.joint_names)
        for joint_id, joint_angle in zip(self.joint_ids, joint_angles):
            pb.resetJointState(
                self._robot_id,
                joint_id,
                joint_angle,
                targetVelocity=0.0,
                physicsClientId=self._client,
            )

    def solve_ik(self, target_pos):
        assert len(target_pos) == 3
        return self._kin_solver.solve_inverse_kinematics(
            target_pos,
            self.joint_angles(),
            self._tinyfk_endeffector_id,
            self._tinyfk_joint_ids,
            with_base=False,
        )

    def set_box(self, pos):
        if self._box_id is not None:
            pb.removeBody(self._box_id)
        vis_box_id = pb.createVisualShape(
            pb.GEOM_BOX,
            halfExtents=[0.05, 0.05, 0.05],
            rgbaColor=[0.0, 1.0, 0, 0.7],
            physicsClientId=self._client,
        )
        box_id = pb.createMultiBody(basePosition=pos, baseVisualShapeIndex=vis_box_id)
        self._box_id = box_id

    def take_photo(self, resolution=1024):
        viewMatrix = pb.computeViewMatrix(
            cameraEyePosition=[1.0, -2.0, 2.5],
            cameraTargetPosition=[0.3, 0, 0],
            cameraUpVector=[0, 1, 0],
        )

        near = 0.01
        far = 5.1
        projectionMatrix = pb.computeProjectionMatrixFOV(
            fov=45.0, aspect=1.0, nearVal=near, farVal=far
        )

        width, height, rgba_, depth_, _ = pb.getCameraImage(
            width=resolution,
            height=resolution,
            viewMatrix=viewMatrix,
            projectionMatrix=projectionMatrix,
        )

        # https://github.com/bulletphysics/bullet3/blob/267f983498c5a249838cd614a01c627b3adda293/examples/pybullet/examples/getCameraImageTest.py#L49
        depth_ = far * near / (far - (far - near) * depth_)
        # depth = 2 * far * near / (far + near - (far - near) * (2 * depth - 1))

        rgb = RGBImage(rgba_[:, :, :3])
        depth = DepthImage(np.expand_dims(depth_, axis=2))
        return rgb, depth

    def get_reachable_target_pos_and_av(self):
        while True:
            try:
                target_pos = np.array([0.5, 0.0, 0.3]) + np.random.randn(3) * np.array(
                    [0.2, 0.25, 0.1]
                )
                angles_solved = self.solve_ik(target_pos)
                break
            except tinyfk._inverse_kinematics.IKFail:
                pass
        return target_pos, angles_solved

    def kinematic_simulate(self, joint_angles_target, N=100, n_pixel=112):
        N_rand = N + np.random.randint(10)
        angles_now = np.array(self.joint_angles())
        step = (np.array(joint_angles_target) - angles_now) / (N_rand - 1)
        angles_list = [AngleVector(angles_now + step * i) for i in range(N_rand)]
        rgb_list = []
        depth_list = []

        for av in angles_list:
            self.set_joint_angles(av)
            rgb, depth = self.take_photo(n_pixel)
            rgb_list.append(rgb)
            depth_list.append(depth)

        n_extend = 30
        rgb_list += [rgb_list[-1]] * n_extend
        depth_list += [depth_list[-1]] * n_extend
        angles_list += [angles_list[-1]] * n_extend

        return ElementSequence(rgb_list), ElementSequence(depth_list), ElementSequence(angles_list)

    def simulate_feedback(self, propagator: Propagator, n_pixel=112) -> List[RGBImage]:
        rgb_list = []
        for i in range(200):
            rgb, depth = self.take_photo(n_pixel)
            rgb_list.append(rgb)

            av = AngleVector(np.array(self.joint_angles()))
            ed = ElementDict([rgb, depth, av])

            propagator.feed(ed)
            av_predicted = propagator.predict(n_prop=1)[0][AngleVector]
            self.set_joint_angles(av_predicted.numpy())

        return rgb_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--feedback", action="store_true", help="feedback mode")
    parser.add_argument("-pn", type=str, default="kuka_reaching", help="project name")
    parser.add_argument("-pp", type=str, help="project path name. preferred over pn.")
    parser.add_argument("-n", type=int, default=100, help="epoch num")
    parser.add_argument("-m", type=int, default=112, help="pixel num")
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

    if project_path_str is None:
        assert project_name is not None
        create_project_dir(project_name)
        project_path = get_project_path(project_name)
    else:
        project_path = Path(project_path_str)
        project_path.mkdir(exist_ok=True)

    np.random.seed(seed)

    if feedback_mode:
        pbdata_path = pybullet_data.getDataPath()
        urdf_path = os.path.join(pbdata_path, "kuka_iiwa", "model.urdf")
        bm = BulletManager(False, urdf_path, "lbr_iiwa_link_7")
        bm.set_joint_angles([0.2 for _ in range(7)])
        target_pos, _ = bm.get_reachable_target_pos_and_av()
        bm.set_box(target_pos)

        # prepare propagator
        propagator = create_default_propagator(project_path)
        if propagator.require_static_context:
            image_encoder = load_default_image_encoder(project_path)
            rgb, _ = bm.take_photo(n_pixel)
            context = image_encoder.forward(rgb)
            propagator.set_static_context(context)

        rgb_list = bm.simulate_feedback(propagator, n_pixel)

        file_path = project_path / "feedback_simulation.gif"
        clip = ImageSequenceClip([rgb.numpy() for rgb in rgb_list], fps=50)
        clip.write_gif(str(file_path), fps=50)
    else:
        with tempfile.TemporaryDirectory() as td:

            def data_generation_task(arg):
                cpu_idx, n_data_gen = arg
                disable_tqdm = cpu_idx != 0

                pbdata_path = pybullet_data.getDataPath()
                urdf_path = os.path.join(pbdata_path, "kuka_iiwa", "model.urdf")
                bm = BulletManager(False, urdf_path, "lbr_iiwa_link_7")

                for i in tqdm.tqdm(range(n_data_gen), disable=disable_tqdm):
                    bm.set_joint_angles([0.2 for _ in range(7)])
                    target_pos, av_solved = bm.get_reachable_target_pos_and_av()
                    bm.set_box(target_pos)
                    rgbimg_seq, dimg_seq, cmd_seq = bm.kinematic_simulate(
                        av_solved, n_pixel=n_pixel
                    )
                    episode_data = EpisodeData.from_seq_list([rgbimg_seq, dimg_seq, cmd_seq])

                    with open(os.path.join(td, str(uuid.uuid4()) + ".pkl"), "wb") as f:
                        pickle.dump(episode_data, f)

            # Because data generation take long, we will use multiple cores if available
            n_cpu = psutil.cpu_count(logical=False)
            print("{} physical cpus are detected".format(n_cpu))

            pool = multiprocessing.Pool(n_cpu)
            n_process_list_assign = [len(lst) for lst in np.array_split(range(n_epoch), n_cpu)]
            pool.map(data_generation_task, zip(range(n_cpu), n_process_list_assign))

            # Collect data and dump bundle of them
            data_list = []
            for file_name in os.listdir(td):
                with open(os.path.join(td, file_name), "rb") as f:
                    data_list.append(pickle.load(f))
            bundle = EpisodeBundle.from_episodes(data_list, n_untouch_episode=n_untouch)
            bundle.dump(project_path)
            bundle.plot_vector_histories(AngleVector, project_path)

            # For debugging
            img_seq = bundle[0].get_sequence_by_type(RGBImage)
            file_path = project_path / "sample.gif"
            clip = ImageSequenceClip([img for img in img_seq], fps=50)
            clip.write_gif(str(file_path), fps=50)
