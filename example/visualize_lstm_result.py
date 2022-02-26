import argparse
import os
from typing import Type

from moviepy.editor import ImageSequenceClip
import matplotlib.pyplot as plt
import numpy as np
import PIL

from mohou.embedder import IdenticalEmbedder
from mohou.embedding_rule import create_embedding_rule
from mohou.file import get_subproject_dir
from mohou.propagator import Propagator
from mohou.trainer import TrainCache
from mohou.types import ElementDict, MultiEpisodeChunk
from mohou.types import AngleVector, ImageBase, get_element_type
from mohou.model import AutoEncoder, LSTM


def add_text_to_image(image: ImageBase, text: str, color: str):

    def canvas_to_ndarray(fig, resize_pixel=None):
        data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        if resize_pixel is None:
            return data
        img = PIL.Image.fromarray(data)
        img_resized = img.resize(resize_pixel)
        data_resized = np.asarray(img_resized)
        return data_resized

    fig, ax = plt.subplots()
    ax.imshow(image.to_rgb()._data)
    ax.text(7, 1, text, fontsize=25, color=color, verticalalignment='top')
    fig.canvas.draw()
    fig.canvas.flush_events()
    return canvas_to_ndarray(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-pn', type=str, default='kuka_reaching', help='project name')
    parser.add_argument('-image', type=str, default='RGBImage', help='image type')

    args = parser.parse_args()
    project_name = args.pn
    image_type: Type[ImageBase] = get_element_type(args.image)  # type: ignore

    chunk = MultiEpisodeChunk.load(project_name).get_intact_chunk()

    tcache_autoencoder = TrainCache.load(project_name, AutoEncoder)
    tcach_lstm = TrainCache.load(project_name, LSTM)
    image_embed_func = tcache_autoencoder.best_model.get_embedder()

    av_idendical_func = IdenticalEmbedder(AngleVector, chunk.get_element_shape(AngleVector)[0])
    embed_rule = create_embedding_rule(image_embed_func, av_idendical_func)

    propagator = Propagator(tcach_lstm.best_model, embed_rule)

    episode_data = chunk[0]
    n_feed = 10
    fed_avs = episode_data.filter_by_type(AngleVector)[:n_feed]
    fed_images = episode_data.filter_by_type(image_type)[:n_feed]

    print("start lstm propagation")
    for elem_tuple in zip(fed_avs, fed_images):
        propagator.feed(ElementDict(elem_tuple))
    print("finish lstm propagation")

    elem_dict_list = propagator.predict(150)
    pred_images = [elem_dict[image_type] for elem_dict in elem_dict_list]

    print("adding text to images...")
    fed_images_with_text = [add_text_to_image(image, 'fed (original)', 'blue') for image in fed_images]
    pred_images_with_text = [add_text_to_image(image, 'predicted by lstm', 'green') for image in pred_images]

    images_with_text = fed_images_with_text + pred_images_with_text

    save_dir = get_subproject_dir(project_name, 'lstm_result')
    full_file_name = os.path.join(save_dir, 'result.gif')
    clip = ImageSequenceClip(images_with_text, fps=20)
    clip.write_gif(full_file_name, fps=20)
