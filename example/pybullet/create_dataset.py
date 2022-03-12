#!usr/bin/env python3
import argparse
import os
import pickle
import multiprocessing
import uuid
import tempfile

import psutil
import tqdm
import numpy as np
from moviepy.editor import ImageSequenceClip
from mohou.file import get_project_dir
from mohou.types import RGBImage, DepthImage, AngleVector
from mohou.types import ElementSequence, EpisodeData, MultiEpisodeChunk

from kinematic_simulator import KinematicBulletSimulator


def run_single_episode(kbsim: KinematicBulletSimulator, n_pixel: int = 112):
    kbsim.set_joint_angles([0.2 for _ in range(7)])
    target_pos, angles_target = kbsim.get_reachable_target_pos_and_av()
    kbsim.set_box(target_pos)

    N_rand = 100 + np.random.randint(10)
    angles_now = np.array(kbsim.joint_angles())
    step = (np.array(angles_target) - angles_now) / (N_rand - 1)
    angles_seq = [angles_now + step * i for i in range(N_rand)]

    rgbarr_seq = []
    deptharr_seq = []
    for angles in angles_seq:
        kbsim.set_joint_angles(angles)
        rgb, depth = kbsim.take_photo(n_pixel)
        rgbarr_seq.append(rgb)
        deptharr_seq.append(depth)

    for i in range(10):
        rgbarr_seq.append(rgbarr_seq[-1])
        deptharr_seq.append(deptharr_seq[-1])
        angles_seq.append(angles_seq[-1])

    return rgbarr_seq, deptharr_seq, angles_seq


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-pn', type=str, default='kuka_reaching', help='project name')
    parser.add_argument('-n', type=int, default=100, help='epoch num')
    parser.add_argument('-m', type=int, default=112, help='pixel num')  # same as mnist
    args = parser.parse_args()
    project_name = args.pn
    n_epoch = args.n
    n_pixel = args.m

    with tempfile.TemporaryDirectory() as td:

        def data_generation_task(arg):
            cpu_idx, n_data_gen = arg
            disable_tqdm = (cpu_idx != 0)  # show progress bar only in a single process

            kbsim = KinematicBulletSimulator()

            for i in tqdm.tqdm(range(n_data_gen), disable=disable_tqdm):
                data = run_single_episode(kbsim, n_pixel)

                with open(os.path.join(td, str(uuid.uuid4()) + '.pkl'), 'wb') as f:
                    pickle.dump(data, f)

        # Because data generation take long, we will use multiple cores if available
        n_cpu = psutil.cpu_count(logical=False)
        print('{} physical cpus are detected'.format(n_cpu))

        pool = multiprocessing.Pool(n_cpu)
        n_process_list_assign = [len(lst) for lst in np.array_split(range(n_epoch), n_cpu)]
        pool.map(data_generation_task, zip(range(n_cpu), n_process_list_assign))

        # Collect data and dump chunk of them
        data_list = []
        for file_name in os.listdir(td):
            with open(os.path.join(td, file_name), 'rb') as f:
                rgbarr_seq, deptharr_seq, angles_seq = pickle.load(f)

            rgb_seq = ElementSequence[RGBImage]()
            depth_seq = ElementSequence[DepthImage]()
            av_seq = ElementSequence[AngleVector]()

            for rgbarr, deptharr, angles in zip(rgbarr_seq, deptharr_seq, angles_seq):
                rgb_seq.append(RGBImage(rgbarr))
                depth_seq.append(DepthImage(deptharr))
                av_seq.append(AngleVector(angles))
            episode_data = EpisodeData((rgb_seq, depth_seq, av_seq))

            data_list.append(episode_data)

        chunk = MultiEpisodeChunk(data_list)
        chunk.dump(project_name)

        # For debugging
        img_seq = chunk[0].filter_by_type(RGBImage)
        filename = os.path.join(get_project_dir(project_name), "sample.gif")
        clip = ImageSequenceClip([img for img in img_seq], fps=50)
        clip.write_gif(filename, fps=50)
