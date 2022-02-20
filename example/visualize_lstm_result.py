import argparse

from mohou.propagator import Propagator
from mohou.trainer import TrainCache
from mohou.types import ElementDict, MultiEpisodeChunk
from mohou.types import AngleVector, RGBImage
from mohou.model import AutoEncoder, LSTM
from mohou.embedder import RGBImageEmbedder, AngleVectorIdenticalEmbedder
from mohou.embedding_rule import RGBAngelVectorEmbeddingRule

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-pn', type=str, default='kuka_reaching', help='project name')

    args = parser.parse_args()

    project_name = args.pn

    chunk = MultiEpisodeChunk.load(project_name)

    tcache_autoencoder = TrainCache.load(project_name, AutoEncoder)
    tcach_lstm = TrainCache.load(project_name, LSTM)
    image_embed_func = tcache_autoencoder.best_model.get_embedder(RGBImageEmbedder)

    av_idendical_func = AngleVectorIdenticalEmbedder(chunk.get_element_shape(AngleVector)[0])
    embed_rule = RGBAngelVectorEmbeddingRule(image_embed_func, av_idendical_func)

    propagator = Propagator(tcach_lstm.best_model, embed_rule)

    episode_data = chunk[0]
    n_feed = 10
    av_seq = episode_data.filter_by_type(AngleVector)[:n_feed]
    iamge_seq = episode_data.filter_by_type(RGBImage)[:n_feed]

    for elem_tuple in zip(av_seq, iamge_seq):
        propagator.feed(ElementDict(elem_tuple))

    elem_dict_list = propagator.predict(100)
