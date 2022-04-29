import numpy as np
from typing import Type, List, Dict

from mohou.embedder import EmbedderBase
from mohou.types import ElementBase, EpisodeData, MultiEpisodeChunk, ElementDict, PrimitiveElementBase, CompositeImageBase
from mohou.utils import assert_with_message


class EmbeddingRule(Dict[Type[ElementBase], EmbedderBase]):

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

    def apply_to_episode_data(self, episode_data: EpisodeData) -> np.ndarray:

        def embed(elem_type, embedder) -> np.ndarray:
            sequence = episode_data.get_sequence_by_type(elem_type)
            return np.stack([embedder.forward(e) for e in sequence])

        vector_seq = np.hstack([embed(k, v) for k, v in self.items()])
        assert_with_message(vector_seq.ndim, 2, 'vector_seq dim')
        return vector_seq

    def apply_to_multi_episode_chunk(self, chunk: MultiEpisodeChunk) -> List[np.ndarray]:

        # TODO(HiroIshida) check chunk compatibility
        def elem_types_to_primitive_elem_set(elem_type_list: List[Type[ElementBase]]):
            primitve_elem_type_list = []
            for elem_type in elem_type_list:
                if issubclass(elem_type, PrimitiveElementBase):
                    primitve_elem_type_list.append(elem_type)
                elif issubclass(elem_type, CompositeImageBase):
                    primitve_elem_type_list.extend(elem_type.image_types)
            return set(primitve_elem_type_list)

        chunk_elem_types = elem_types_to_primitive_elem_set(list(chunk.types()))
        required_elem_types = elem_types_to_primitive_elem_set(list(self.keys()))
        assert required_elem_types <= chunk_elem_types

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

    @classmethod
    def from_embedders(cls, embedder_list: List[EmbedderBase]) -> 'EmbeddingRule':
        rule = cls()
        for embedder in embedder_list:
            rule[embedder.elem_type] = embedder
        return rule
