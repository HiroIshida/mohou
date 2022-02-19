import numpy as np
from typing import Type, List, OrderedDict

from mohou.embedding_functor import EmbeddingFunctor, ImageEmbeddingFunctor, IdenticalEmbeddingFunctor
from mohou.types import ElementBase, ElementSequence, EpisodeData, MultiEpisodeChunk
from mohou.types import AngleVector, RGBImage


class EmbeddingRule(OrderedDict[Type[ElementBase], EmbeddingFunctor]):

    def apply(self, elements: List[ElementBase]) -> np.ndarray:

        def embed(elem_type, reducer) -> np.ndarray:
            for e in elements:
                if isinstance(e, elem_type):
                    return reducer(e)
            assert False

        vector = np.hstack([embed(k, v) for k, v in self.items()])
        return vector

    def apply_to_episode_data(self, episode_data: EpisodeData) -> np.ndarray:

        def embed(elem_type, reducer) -> np.ndarray:
            for sequence in episode_data:
                if isinstance(sequence[0], elem_type):
                    return np.concatenate([reducer(e) for e in sequence])
            assert False

        vectors = np.hstack([embed(k, v) for k, v in self.items()])
        return vectors

    @property
    def dimension(self) -> int:
        return sum(reducer.output_size for reducer in self.values())

    def __str__(self) -> str:
        string = 'total dim: {}'.format(self.dimension)
        for elem_type, reducer in self.items():
            string += '\n{0}: {1}'.format(elem_type.__name__, reducer.output_size)
        return string


class RGBAngelVectorEmbeddingRule(EmbeddingRule):

    def __init__(self, image_reducer: ImageEmbeddingFunctor, identical_reducer: IdenticalEmbeddingFunctor):
        self[RGBImage] = image_reducer
        self[AngleVector] = identical_reducer
