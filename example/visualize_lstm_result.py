import argparse

from mohou.default import create_default_propagator
from mohou.types import MultiEpisodeChunk, AngleVector
from mohou.script_utils import auto_detect_autoencoder_type
from mohou.script_utils import visualize_lstm_propagation

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-pn', type=str, default='kuka_reaching', help='project name')
    parser.add_argument('-n', type=int, default=150, help='number of visualization')

    args = parser.parse_args()
    project_name = args.pn
    n_prop = args.n

    chunk_spec = MultiEpisodeChunk.load_spec(project_name)
    n_av_dim = chunk_spec.type_shape_table[AngleVector][0]
    ae_type = auto_detect_autoencoder_type(project_name)
    propagator = create_default_propagator(project_name, n_av_dim, ae_type=ae_type)
    visualize_lstm_propagation(project_name, propagator, n_prop)
