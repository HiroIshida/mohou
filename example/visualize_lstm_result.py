import argparse
from typing import Type

from mohou.default import create_default_propagator
from mohou.types import MultiEpisodeChunk
from mohou.types import ImageBase, AngleVector, get_element_type
from mohou.script_utils import visualize_lstm_propagation


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-pn', type=str, default='kuka_reaching', help='project name')
    parser.add_argument('-n', type=int, default=150, help='number of visualization')
    parser.add_argument('-image', type=str, default='RGBImage', help='image type')

    args = parser.parse_args()
    project_name = args.pn
    n_prop = args.n
    image_type: Type[ImageBase] = get_element_type(args.image)  # type: ignore

    chunk_spec = MultiEpisodeChunk.load_spec(project_name)
    n_av_dim = chunk_spec.type_shape_table[AngleVector][0]
    propagator = create_default_propagator(project_name, n_av_dim)
    visualize_lstm_propagation(project_name, propagator, image_type, n_prop)
