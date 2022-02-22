import argparse
import os

from moviepy.editor import ImageSequenceClip

from mohou.embedder import IdenticalEmbedder
from mohou.embedding_rule import RGBAngelVectorEmbeddingRule
from mohou.file import get_subproject_dir
from mohou.propagator import Propagator
from mohou.trainer import TrainCache
from mohou.types import ElementDict, MultiEpisodeChunk
from mohou.types import AngleVector, RGBImage
from mohou.model import AutoEncoder, LSTM

from utils import add_text_to_image

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-pn', type=str, default='kuka_reaching', help='project name')

    args = parser.parse_args()

    project_name = args.pn

    chunk = MultiEpisodeChunk.load(project_name).get_intact_chunk()

    tcache_autoencoder = TrainCache.load(project_name, AutoEncoder)
    tcach_lstm = TrainCache.load(project_name, LSTM)
    image_embed_func = tcache_autoencoder.best_model.get_embedder()

    av_idendical_func = IdenticalEmbedder(AngleVector, chunk.get_element_shape(AngleVector)[0])
    embed_rule = RGBAngelVectorEmbeddingRule(image_embed_func, av_idendical_func)

    propagator = Propagator(tcach_lstm.best_model, embed_rule)

    episode_data = chunk[0]
    n_feed = 10
    fed_avs = episode_data.filter_by_type(AngleVector)[:n_feed]
    fed_images = episode_data.filter_by_type(RGBImage)[:n_feed]

    print("start lstm propagation")
    for elem_tuple in zip(fed_avs, fed_images):
        propagator.feed(ElementDict(elem_tuple))
    print("finish lstm propagation")

    elem_dict_list = propagator.predict(150)
    pred_images = [elem_dict[RGBImage] for elem_dict in elem_dict_list]

    print("adding text to images...")
    fed_images_with_text = [add_text_to_image(im, 'fed (original)', 'blue') for im in fed_images]
    pred_images_with_text = [add_text_to_image(im, 'predicted by lstm', 'green') for im in pred_images]

    images_with_text = fed_images_with_text + pred_images_with_text

    save_dir = get_subproject_dir(project_name, 'lstm_result')
    full_file_name = os.path.join(save_dir, 'result.gif')
    clip = ImageSequenceClip(images_with_text, fps=20)
    clip.write_gif(full_file_name, fps=20)
