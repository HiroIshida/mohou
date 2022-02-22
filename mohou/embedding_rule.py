import numpy as np
from typing import Type, List, Dict

from mohou.embedder import Embedder, ImageEmbedder, IdenticalEmbedder
from mohou.types import ElementBase, EpisodeData, MultiEpisodeChunk, ElementDict
from mohou.types import AngleVector, RGBImage, RGBDImage


class EmbeddingRule(Dict[Type[ElementBase], Embedder]):

    def apply(self, elem_dict: ElementDict) -> np.ndarray:
        vector_list = []
        for elem_type, embedder in self.items():
            vector = embedder.forward(elem_dict[elem_type])
            vector_list.append(vector)
        return np.hstack(vector_list)

    def inverse_apply(self, vector: np.ndarray) -> ElementDict:

        def split_vector(vector: np.ndarray, size_list: List[int]):
            head = 0
            vector_list = []
            for i, size in enumerate(size_list):
                tail = head + size
                vector_list.append(vector[head:tail])
                head = tail
            return vector_list

        size_list = [embedder.output_size for elem_type, embedder in self.items()]
        vector_list = split_vector(vector, size_list)

        elem_dict = ElementDict([])
        for vec, (elem_type, embedder) in zip(vector_list, self.items()):
            elem_dict[elem_type] = embedder.backward(vec)
        return elem_dict

    def apply_to_list(self, elements: List[ElementBase]) -> np.ndarray:

        def embed(elem_type, embedder) -> np.ndarray:
            for e in elements:
                if isinstance(e, elem_type):
                    return embedder.forward(e)
            assert False

        vector = np.hstack([embed(k, v) for k, v in self.items()])
        return vector

    def apply_to_episode_data(self, episode_data: EpisodeData) -> np.ndarray:

        def embed(elem_type, embedder) -> np.ndarray:
            for sequence in episode_data:
                if isinstance(sequence[0], elem_type):
                    return np.stack([embedder.forward(e) for e in sequence])
            assert False

        vector_seq = np.hstack([embed(k, v) for k, v in self.items()])

        assert vector_seq.ndim == 2
        return vector_seq

    def apply_to_multi_episode_chunk(self, chunk: MultiEpisodeChunk) -> List[np.ndarray]:

        assert set(self.keys()) <= set(chunk.type_shape_table.keys())

        vector_seq_list = [self.apply_to_episode_data(data) for data in chunk]

        assert vector_seq_list[0].ndim == 2
        return vector_seq_list

    @property
    def dimension(self) -> int:
        return sum(embedder.output_size for embedder in self.values())

    def __str__(self) -> str:
        string = 'total dim: {}'.format(self.dimension)
        for elem_type, embedder in self.items():
            string += '\n{0}: {1}'.format(elem_type.__name__, embedder.output_size)
        return string


class RGBAngelVectorEmbeddingRule(EmbeddingRule):

    def __init__(self, image_embedder: ImageEmbedder, identical_embedder: IdenticalEmbedder):
        self[RGBImage] = image_embedder
        self[AngleVector] = identical_embedder


class RGBDAngelVectorEmbeddingRule(EmbeddingRule):

    def __init__(self, image_embedder: ImageEmbedder, identical_embedder: IdenticalEmbedder):
        self[RGBDImage] = image_embedder
        self[AngleVector] = identical_embedder
