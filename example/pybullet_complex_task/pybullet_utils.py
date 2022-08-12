import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple
from xml.etree.ElementTree import Element, ElementTree, SubElement

import numpy as np
import pybullet as pb
from skrobot.coordinates import Coordinates


@dataclass
class BoxConfig:
    size: Tuple[float, float, float]
    rgba: Tuple[float, float, float, float] = (1, 1, 1, 1)
    density: float = 1.0 * 10**3
    name: str = "box"

    @property
    def mass(self):
        v = np.prod(self.size)
        return v * self.density

    @property
    def inertia(self):
        a, b, c = self.size
        ixx = 1 / 3.0 * (b**2 + c**2) * self.mass
        iyy = 1 / 3.0 * (c**2 + a**2) * self.mass
        izz = 1 / 3.0 * (a**2 + b**2) * self.mass
        return ixx, iyy, izz


def create_box_urdf(config: BoxConfig) -> Path:
    root = Element("robot", name=config.name)

    # create material
    material = SubElement(root, "material", name="material_name")
    SubElement(material, "color", rgba="{} {} {} {}".format(*config.rgba))

    link = SubElement(root, "link", name="base_link")

    # create geometry (a part of both visual and collision)
    geometry = Element("geometry")
    SubElement(geometry, "box", size="{} {} {}".format(*config.size))

    # create visual
    visual = Element("visual")
    visual.append(geometry)
    SubElement(visual, "material", name="material_name")
    link.append(visual)

    # create collision
    collision = Element("collision")
    collision.append(geometry)
    link.append(collision)

    # create inertial
    inertial = Element("inertial")
    SubElement(inertial, "origin", rpy="0 0 0", xyz="0 0 0")
    SubElement(inertial, "mass", value=str(config.mass))
    ixx, iyy, izz = config.inertia
    SubElement(
        inertial,
        "inertia",
        ixx=str(ixx),
        ixy="0.0",
        ixz="0.0",
        iyy=str(iyy),
        iyz="0.0",
        izz=str(izz),
    )
    link.append(inertial)

    tree = ElementTree(root)
    directory_path = Path("/tmp/auto_generated_urdf")
    directory_path.mkdir(exist_ok=True)
    file_path = directory_path / "{}.urdf".format(uuid.uuid4())
    tree.write(str(file_path))
    return file_path


def create_box(config: BoxConfig, friction: float = 0.7) -> int:
    p = create_box_urdf(config)
    obj_id = pb.loadURDF(str(p))
    pb.changeDynamics(obj_id, -1, lateralFriction=friction)
    return obj_id


def create_debug_axis(coords: Coordinates, length: float = 0.1):
    # pb.removeAllUserDebugItems()
    start = coords.worldpos()
    end_x = start + coords.rotate_vector([length, 0, 0])
    pb.addUserDebugLine(start, end_x, [1, 0, 0], 3)
    end_y = start + coords.rotate_vector([0, length, 0])
    pb.addUserDebugLine(start, end_y, [0, 1, 0], 3)
    end_z = start + coords.rotate_vector([0, 0, length])
    pb.addUserDebugLine(start, end_z, [0, 0, 1], 3)
