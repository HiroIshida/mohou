import argparse
import os
import random

import matplotlib.pyplot as plt
import torchvision

from mohou.dataset import RGBAutoEncoderDataset
from mohou.file import get_subproject_dir
from mohou.model import AutoEncoder
from mohou.trainer import TrainCache
from mohou.types import MultiEpisodeChunk


def debug_visualize_reconstruction(
        project_name: str, dataset: RGBAutoEncoderDataset, tcache: TrainCache, n_vis: int = 5):

    idxes = list(range(len(dataset)))
    random.shuffle(idxes)
    idxes_test = idxes[:min(n_vis, len(dataset))]

    for i, idx in enumerate(idxes_test):

        image_torch = dataset[idx].unsqueeze(dim=0)
        image_torch_reconstructed = tcache.best_model(image_torch)

        to_pil_image = torchvision.transforms.ToPILImage()
        image = to_pil_image(image_torch.squeeze())
        image_reconstructed = to_pil_image(image_torch_reconstructed.squeeze())

        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.suptitle('left: original, right: reconstructed')
        ax1.imshow(image)
        ax2.imshow(image_reconstructed)
        save_dir = get_subproject_dir(project_name, 'autoencoder_result')

        full_file_name = os.path.join(save_dir, 'result{}.png'.format(i))
        plt.savefig(full_file_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-pn', type=str, default='kuka_reaching', help='project name')
    parser.add_argument('-n', type=int, default=5, help='number of visualization')
    args = parser.parse_args()
    project_name = args.pn
    n_vis = args.n

    chunk = MultiEpisodeChunk.load(project_name)
    dataset = RGBAutoEncoderDataset.from_chunk(chunk)
    tcache = TrainCache.load(project_name, AutoEncoder)
    debug_visualize_reconstruction(project_name, dataset, tcache, n_vis)
