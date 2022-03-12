#!/usr/bin/env python3
import argparse
import os
from typing import List

from moviepy.editor import ImageSequenceClip
from mohou.file import get_project_dir
from mohou.types import RGBImage, DepthImage, AngleVector, ElementDict
from mohou.model import AutoEncoder
from mohou.trainer import TrainCache
from mohou.propagator import Propagator, create_default_propagator

from kinematic_simulator import KinematicBulletSimulator


def simulate_feedback(
        kbsim: KinematicBulletSimulator,
        propagator: Propagator,
        n_pixel=112) -> List[RGBImage]:

    rgb_list_debug_gif = []
    for i in range(120):
        rgbarr, deptharr = kbsim.take_photo(n_pixel)
        angles = kbsim.joint_angles()

        ed = ElementDict([RGBImage(rgbarr), DepthImage(deptharr), AngleVector(angles)])
        propagator.feed(ed)
        av_predicted = propagator.predict(n_prop=1)[0][AngleVector]

        kbsim.set_joint_angles(av_predicted.numpy())

        rgb_list_debug_gif.append(rgbarr)

    return rgb_list_debug_gif


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-pn', type=str, default='kuka_reaching', help='project name')
    args = parser.parse_args()
    project_name = args.pn

    propagator = create_default_propagator(project_name, 7)
    kbsim = KinematicBulletSimulator()
    n_pixel = TrainCache.load(project_name, AutoEncoder).best_model.config.n_pixel
    rgb_list_debug_gif = simulate_feedback(kbsim, propagator, n_pixel=n_pixel)

    filename = os.path.join(get_project_dir(project_name), "feedback_simulation.gif")
    clip = ImageSequenceClip(rgb_list_debug_gif, fps=50)
    clip.write_gif(filename, fps=50)
