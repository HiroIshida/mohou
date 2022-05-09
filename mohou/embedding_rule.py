from abc import ABC, abstractmethod
import copy

import functools  # for cached_property
if hasattr(functools, 'cached_property'):
    from functools import cached_property
else:
    from cached_property import cached_property  # type: ignore

from dataclasses import dataclass
import numpy as np
from typing import Type, List, Dict, Generator, Optional

from mohou.embedder import EmbedderBase
from mohou.types import ElementBase, EpisodeData, MultiEpisodeChunk, ElementDict, PrimitiveElementBase, CompositeImageBase
from mohou.utils import assert_with_message


class PostProcessor(ABC):

    @abstractmethod
    def apply(self, vec: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def inverse_apply(self, vec: np.ndarray) -> np.ndarray:
        pass


class IdenticalPostProcessor(PostProcessor):

    def apply(self, vec: np.ndarray) -> np.ndarray:
        return vec

    def inverse_apply(self, vec: np.ndarray) -> np.ndarray:
        return vec


@dataclass(frozen=True)
class ElemCovMatchPostProcessor(PostProcessor):
    dims: List[int]
    means: List[np.ndarray]
    covs: List[np.ndarray]

    @staticmethod
    def get_ranges(dims: List[int]) -> Generator[slice, None, None]:
        head = 0
        for dim in dims:
            yield slice(head, head + dim)
            head += dim

    @cached_property
    def characteristic_stds(self) -> np.ndarray:

        def get_max_std(cov) -> float:
            eig_values, _ = np.linalg.eig(cov)
            max_eig_cov = max(eig_values)
            return np.sqrt(max_eig_cov)

        char_stds = np.array(list(map(get_max_std, self.covs)))
        return char_stds

    @cached_property
    def scaled_characteristic_stds(self) -> np.ndarray:
        c_stds = self.characteristic_stds
        return c_stds / np.max(c_stds)

    @classmethod
    def from_feature_seqs(cls, feature_seq: np.ndarray, dims: List[int]):
        means = []
        covs = []
        for rang in cls.get_ranges(dims):
            feature_seq_partial = feature_seq[:, rang]
            means.append(np.mean(feature_seq_partial))
            covs.append(np.cov(feature_seq_partial.T))
        return cls(dims, means, covs)

    def apply(self, vec: np.ndarray) -> np.ndarray:
        vec_out = copy.deepcopy(vec)
        char_stds = self.scaled_characteristic_stds
        for idx_elem, rangee in enumerate(self.get_ranges(self.dims)):
            vec_out_new = (vec_out[rangee] - self.means[idx_elem]) * char_stds[idx_elem]  # type: ignore
            vec_out[rangee] = vec_out_new
        return vec_out

    def inverse_apply(self, vec: np.ndarray) -> np.ndarray:
        vec_out = copy.deepcopy(vec)
        char_stds = self.scaled_characteristic_stds
        for idx_elem, rangee in enumerate(self.get_ranges(self.dims)):
            vec_out_new = (vec_out[rangee] / char_stds[idx_elem]) + self.means[idx_elem]
            vec_out[rangee] = vec_out_new
        return vec_out


class EmbeddingRule(Dict[Type[ElementBase], EmbedderBase]):
    post_processor: PostProcessor

    def apply(self, elem_dict: ElementDict) -> np.ndarray:
        vector_list = []
        for elem_type, embedder in self.items():
            vector = embedder.forward(elem_dict[elem_type])
            vector_processed = self.post_processor.apply(vector)
            vector_list.append(vector_processed)
        return np.hstack(vector_list)

    def inverse_apply(self, vector_processed: np.ndarray) -> ElementDict:

        def split_vector(vector: np.ndarray, size_list: List[int]):
            head = 0
            vector_list = []
            for i, size in enumerate(size_list):
                tail = head + size
                vector_list.append(vector[head:tail])
                head = tail
            return vector_list

        vector = self.post_processor.inverse_apply(vector_processed)
        size_list = [embedder.output_size for elem_type, embedder in self.items()]
        vector_list = split_vector(vector, size_list)

        elem_dict = ElementDict([])
        for vec, (elem_type, embedder) in zip(vector_list, self.items()):
            elem_dict[elem_type] = embedder.backward(vec)
        return elem_dict

    def apply_to_episode_data(self, episode_data: EpisodeData) -> np.ndarray:

        def encode_and_postprocess(elem_type, embedder) -> np.ndarray:
            sequence = episode_data.get_sequence_by_type(elem_type)
            vector = np.stack([embedder.forward(e) for e in sequence])
            vector_processed = self.post_processor.apply(vector)
            return vector_processed

        vector_seq = np.hstack([encode_and_postprocess(k, v) for k, v in self.items()])
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
    def from_embedders(cls, embedder_list: List[EmbedderBase], chunk: Optional[MultiEpisodeChunk] = None) -> 'EmbeddingRule':
        rule: EmbeddingRule = cls()
        for embedder in embedder_list:
            rule[embedder.elem_type] = embedder
        rule.post_processor = IdenticalPostProcessor()

        if chunk is not None:
            # compute normalizer and set to embedder
            vector_seqs = rule.apply_to_multi_episode_chunk(chunk)
            vector_seq_concated = np.concatenate(vector_seqs, axis=0)
            dims = [emb.output_size for emb in embedder_list]
            normalizer = ElemCovMatchPostProcessor.from_feature_seqs(vector_seq_concated, dims)
            rule.post_processor = normalizer
        return rule
