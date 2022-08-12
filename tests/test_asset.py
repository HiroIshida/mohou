from tinyfk import RobotModel

from mohou.asset import get_panda_urdf_path


def test_panda_urdf_path():
    path = get_panda_urdf_path()
    RobotModel(str(path))
