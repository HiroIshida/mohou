import os

import tinyfk
import numpy as np
import pybullet as pb
import pybullet_data


class KinematicBulletSimulator(object):

    def __init__(self, use_gui=False):
        pbdata_path = pybullet_data.getDataPath()
        urdf_path = os.path.join(pbdata_path, 'kuka_iiwa', 'model.urdf')
        end_effector_name = 'lbr_iiwa_link_7'

        client = pb.connect(pb.GUI if use_gui else pb.DIRECT)
        pb.configureDebugVisualizer(pb.COV_ENABLE_GUI, 0)
        robot_id = pb.loadURDF(urdf_path)
        pbdata_path = pybullet_data.getDataPath()
        pb.loadURDF(os.path.join(pbdata_path, "samurai.urdf"))

        link_table = {pb.getBodyInfo(robot_id, physicsClientId=client)[0].decode('UTF-8'): -1}
        joint_table = {}

        def heck(path):
            return "_".join(path.split("/"))

        for _id in range(pb.getNumJoints(robot_id, physicsClientId=client)):
            joint_info = pb.getJointInfo(robot_id, _id, physicsClientId=client)
            joint_id = joint_info[0]
            joint_name = joint_info[1].decode('UTF-8')
            joint_table[joint_name] = joint_id
            name_ = joint_info[12].decode('UTF-8')
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
        return np.array([
            pb.getJointState(self._robot_id, joint_id, physicsClientId=self._client)[0]
            for joint_id in self.joint_ids])

    def set_joint_angles(self, joint_angles):
        assert len(joint_angles) == len(self.joint_names)
        for joint_id, joint_angle in zip(self.joint_ids, joint_angles):
            pb.resetJointState(self._robot_id, joint_id, joint_angle,
                               targetVelocity=0.0,
                               physicsClientId=self._client)

    def solve_ik(self, target_pos):
        assert len(target_pos) == 3
        return self._kin_solver.solve_inverse_kinematics(
            target_pos,
            self.joint_angles(),
            self._tinyfk_endeffector_id,
            self._tinyfk_joint_ids,
            with_base=False)

    def set_box(self, pos):
        if self._box_id is not None:
            pb.removeBody(self._box_id)
        vis_box_id = pb.createVisualShape(
            pb.GEOM_BOX,
            halfExtents=[0.05, 0.05, 0.05],
            rgbaColor=[0.0, 1.0, 0, 0.7],
            physicsClientId=self._client)
        box_id = pb.createMultiBody(basePosition=pos, baseVisualShapeIndex=vis_box_id)
        self._box_id = box_id

    def take_photo(self, resolution=1024):
        viewMatrix = pb.computeViewMatrix(
            cameraEyePosition=[1.0, -2.0, 2.5],
            cameraTargetPosition=[0.3, 0, 0],
            cameraUpVector=[0, 1, 0])

        near = 0.01
        far = 5.1
        projectionMatrix = pb.computeProjectionMatrixFOV(
            fov=45.0,
            aspect=1.0,
            nearVal=near,
            farVal=far)

        width, height, rgba, depth_, _ = pb.getCameraImage(
            width=resolution,
            height=resolution,
            viewMatrix=viewMatrix,
            projectionMatrix=projectionMatrix)

        # https://github.com/bulletphysics/bullet3/blob/267f983498c5a249838cd614a01c627b3adda293/examples/pybullet/examples/getCameraImageTest.py#L49
        rgb = rgba[:, :, :3]
        depth = far * near / (far - (far - near) * depth_)
        return rgb, np.expand_dims(depth, axis=2)

    def get_reachable_target_pos_and_av(self):
        while True:
            try:
                target_pos = np.array([0.5, 0.0, 0.3]) + np.random.randn(3) * np.array([0.2, 0.5, 0.1])
                angles_solved = self.solve_ik(target_pos)
                break
            except tinyfk._inverse_kinematics.IKFail:
                pass
        return target_pos, angles_solved
