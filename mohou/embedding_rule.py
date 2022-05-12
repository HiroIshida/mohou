from abc import ABC, abstractmethod
import copy
import logging

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

logger = logging.getLogger(__name__)


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

    def __post_init__(self):
        for i, dim in enumerate(self.dims):
            assert_with_message(self.means[i].shape, (dim,), 'mean shape of {}'.format(i))
            assert_with_message(self.covs[i].shape, (dim, dim), 'cov shape of {}'.format(i))

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
        logger.info('char stds: {}'.format(char_stds))
        return char_stds

    @cached_property
    def scaled_characteristic_stds(self) -> np.ndarray:
        c_stds = self.characteristic_stds
        return c_stds / np.max(c_stds)

    @staticmethod
    def is_binary_sequence(partial_feature_seq: np.ndarray):
        return len(set(partial_feature_seq.flatten().tolist())) == 2

    @classmethod
    def from_feature_seqs(cls, feature_seq: np.ndarray, dims: List[int]):
        assert_with_message(feature_seq.ndim, 2, 'feature_seq.ndim')
        means = []
        covs = []
        for rang in cls.get_ranges(dims):
            feature_seq_partial = feature_seq[:, rang]
            dim = feature_seq_partial.shape[1]
            if cls.is_binary_sequence(feature_seq_partial):
                # because it's strange to compute covariance for binary sequence
                assert dim == 1, 'this restriction maybe removed'
                minn = np.min(feature_seq_partial)
                maxx = np.max(feature_seq_partial)
                cov = np.diag(np.ones(dim))
                mean = np.array([0.5 * (minn + maxx)])
            else:
                mean = np.mean(feature_seq_partial, axis=0)
                cov = np.cov(feature_seq_partial.T)
                if cov.ndim == 0:  # unfortunately, np.cov return 0 dim array instead of 1x1
                    cov = np.expand_dims(cov, axis=0)
                    cov = np.array([[cov.item()]])
            means.append(mean)
            covs.append(cov)
        return cls(dims, means, covs)

    def apply(self, vec: np.ndarray) -> np.ndarray:
        assert_with_message(vec.ndim, 1, 'vector dim')
        assert_with_message(len(vec), sum(self.dims), 'vector total dim')
        vec_out = copy.deepcopy(vec)
        char_stds = self.scaled_characteristic_stds
        for idx_elem, rangee in enumerate(self.get_ranges(self.dims)):
            vec_out_new = (vec_out[rangee] - self.means[idx_elem]) / char_stds[idx_elem]  # type: ignore
            vec_out[rangee] = vec_out_new
        return vec_out

    def inverse_apply(self, vec: np.ndarray) -> np.ndarray:
        assert_with_message(vec.ndim, 1, 'vector dim')
        assert_with_message(len(vec), sum(self.dims), 'vector total dim')
        vec_out = copy.deepcopy(vec)
        char_stds = self.scaled_characteristic_stds
        for idx_elem, rangee in enumerate(self.get_ranges(self.dims)):
            vec_out_new = (vec_out[rangee] * char_stds[idx_elem]) + self.means[idx_elem]
            vec_out[rangee] = vec_out_new
        return vec_out


class EmbeddingRule(Dict[Type[ElementBase], EmbedderBase]):
    post_processor: PostProcessor

    def apply(self, elem_dict: ElementDict) -> np.ndarray:
        vector_list = []
        for elem_type, embedder in self.items():
            vector = embedder.forward(elem_dict[elem_type])
            vector_list.append(vector)
        return self.post_processor.apply(np.hstack(vector_list))

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
            vectors = [embedder.forward(e) for e in sequence]
            return np.stack(vectors)

        vector_seq = np.hstack([encode_and_postprocess(k, v) for k, v in self.items()])
        vector_seq_processed = np.array([self.post_processor.apply(e) for e in vector_seq])
        assert_with_message(vector_seq_processed.ndim, 2, 'vector_seq dim')
        return vector_seq_processed

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
