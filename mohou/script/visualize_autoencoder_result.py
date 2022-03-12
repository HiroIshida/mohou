import argparse
import os
import random
from typing import Type

import matplotlib.pyplot as plt

from mohou.dataset import AutoEncoderDataset
from mohou.file import get_subproject_dir
from mohou.model import AutoEncoder
from mohou.trainer import TrainCache
from mohou.types import ImageBase, MultiEpisodeChunk, get_element_type


def debug_visualize_reconstruction(
        project_name: str, dataset: AutoEncoderDataset, tcache: TrainCache, n_vis: int = 5):

    idxes = list(range(len(dataset)))
    random.shuffle(idxes)
    idxes_test = idxes[:min(n_vis, len(dataset))]

    for i, idx in enumerate(idxes_test):

        image_torch = dataset[idx].unsqueeze(dim=0)
        image_torch_reconstructed = tcache.best_model(image_torch)

        img = dataset.image_type.from_tensor(image_torch.squeeze(dim=0))
        img_reconstructed = dataset.image_type.from_tensor(image_torch_reconstructed.squeeze(dim=0))

        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.suptitle('left: original, right: reconstructed')
        ax1.imshow(img.to_rgb()._data)
        ax2.imshow(img_reconstructed.to_rgb()._data)
        save_dir = get_subproject_dir(project_name, 'autoencoder_result')

        full_file_name = os.path.join(save_dir, 'result{}.png'.format(i))
        plt.savefig(full_file_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-pn', type=str, default='kuka_reaching', help='project name')
    parser.add_argument('-n', type=int, default=5, help='number of visualization')
    parser.add_argument('-image', type=str, default='RGBImage', help='image type')
    args = parser.parse_args()
    project_name = args.pn
    n_vis = args.n
    image_type: Type[ImageBase] = get_element_type(args.image)  # type: ignore

    chunk = MultiEpisodeChunk.load(project_name).get_intact_chunk()
    dataset = AutoEncoderDataset.from_chunk(chunk, image_type)
    tcache = TrainCache.load(project_name, AutoEncoder)
    debug_visualize_reconstruction(project_name, dataset, tcache, n_vis)
