import copy
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Generator, List, Optional, Type

import numpy as np

from mohou.encoder import EncoderBase
from mohou.types import (
    CompositeImageBase,
    ElementBase,
    ElementDict,
    EpisodeData,
    MultiEpisodeChunk,
    PrimitiveElementBase,
)
from mohou.utils import assert_equal_with_message

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


@dataclass
class ElemCovMatchPostProcessor(PostProcessor):
    @dataclass
    class NormalizationSpec:
        dim: int
        mean: np.ndarray
        cov: np.ndarray

    type_spec_table: Dict[Type[ElementBase], NormalizationSpec]
    _scaled_charc_stds: Optional[np.ndarray] = None

    def __post_init__(self):
        for key, spec in self.type_spec_table.items():
            dim = spec.dim
            mean = spec.mean
            cov = spec.cov
            type_name = key.__name__
            assert_equal_with_message(mean.shape, (dim,), "mean shape of {}".format(type_name))
            assert_equal_with_message(cov.shape, (dim, dim), "cov shape of {}".format(type_name))

    def delete(self, elem_type: Type[ElementBase]) -> None:
        self.type_spec_table.pop(elem_type)
        self._scaled_charc_stds = None  # delete cache

    @property
    def dims(self) -> List[int]:
        return [val.dim for val in self.type_spec_table.values()]

    @property
    def means(self) -> List[np.ndarray]:
        return [val.mean for val in self.type_spec_table.values()]

    @property
    def covs(self) -> List[np.ndarray]:
        return [val.cov for val in self.type_spec_table.values()]

    @staticmethod
    def get_ranges(dims: List[int]) -> Generator[slice, None, None]:
        head = 0
        for dim in dims:
            yield slice(head, head + dim)
            head += dim

    def get_characteristic_stds(self) -> np.ndarray:
        def get_max_std(cov) -> float:
            eig_values, _ = np.linalg.eig(cov)
            max_eig_cov = max(eig_values)
            return np.sqrt(max_eig_cov)

        charc_stds = np.array(list(map(get_max_std, self.covs)))
        logger.info("char stds: {}".format(charc_stds))
        return charc_stds

    def get_scaled_characteristic_stds(self) -> np.ndarray:
        # use cache if exists
        if self._scaled_charc_stds is not None:
            return self._scaled_charc_stds

        charc_stds = self.get_characteristic_stds()
        scaled_charc_stds = charc_stds / np.max(charc_stds)
        self._scaled_charc_stds = scaled_charc_stds  # caching
        return scaled_charc_stds

    @staticmethod
    def is_binary_sequence(partial_feature_seq: np.ndarray):
        return len(set(partial_feature_seq.flatten().tolist())) == 2

    @classmethod
    def from_feature_seqs(
        cls, feature_seq: np.ndarray, type_dim_table: Dict[Type[ElementBase], int]
    ):

        Spec = ElemCovMatchPostProcessor.NormalizationSpec

        assert_equal_with_message(feature_seq.ndim, 2, "feature_seq.ndim")
        dims = list(type_dim_table.values())

        type_spec_table = {}

        for typee, rangee in zip(type_dim_table.keys(), cls.get_ranges(dims)):
            feature_seq_partial = feature_seq[:, rangee]
            dim = feature_seq_partial.shape[1]
            if cls.is_binary_sequence(feature_seq_partial):
                # because it's strange to compute covariance for binary sequence
                assert dim == 1, "this restriction maybe removed"
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
            type_spec_table[typee] = Spec(dim, mean, cov)
        return cls(type_spec_table)

    def apply(self, vec: np.ndarray) -> np.ndarray:
        assert_equal_with_message(vec.ndim, 1, "vector dim")
        assert_equal_with_message(len(vec), sum(self.dims), "vector total dim")

        vec_out = copy.deepcopy(vec)
        charc_stds = self.get_scaled_characteristic_stds()
        for idx_elem, rangee in enumerate(self.get_ranges(self.dims)):
            vec_out_new = (vec_out[rangee] - self.means[idx_elem]) / charc_stds[idx_elem]  # type: ignore
            vec_out[rangee] = vec_out_new
        return vec_out

    def inverse_apply(self, vec: np.ndarray) -> np.ndarray:
        assert_equal_with_message(vec.ndim, 1, "vector dim")
        assert_equal_with_message(len(vec), sum(self.dims), "vector total dim")
        vec_out = copy.deepcopy(vec)
        char_stds = self.get_scaled_characteristic_stds()
        for idx_elem, rangee in enumerate(self.get_ranges(self.dims)):
            vec_out_new = (vec_out[rangee] * char_stds[idx_elem]) + self.means[idx_elem]
            vec_out[rangee] = vec_out_new
        return vec_out


class EncodingRule(Dict[Type[ElementBase], EncoderBase]):
    post_processor: PostProcessor

    def pop(self, args):
        # As we have delete function, it is bit confusing
        raise NotImplementedError  # delete this method if Dict

    def delete(self, elem_type: Type[ElementBase]) -> None:
        if isinstance(self.post_processor, ElemCovMatchPostProcessor):
            self.post_processor.delete(elem_type)
        super().pop(elem_type)

    def apply(self, elem_dict: ElementDict) -> np.ndarray:
        vector_list = []
        for elem_type, encoder in self.items():
            vector = encoder.forward(elem_dict[elem_type])
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
        size_list = [encoder.output_size for elem_type, encoder in self.items()]
        vector_list = split_vector(vector, size_list)

        elem_dict = ElementDict([])
        for vec, (elem_type, encoder) in zip(vector_list, self.items()):
            elem_dict[elem_type] = encoder.backward(vec)
        return elem_dict

    def apply_to_episode_data(self, episode_data: EpisodeData) -> np.ndarray:
        def encode_and_postprocess(elem_type, encoder) -> np.ndarray:
            sequence = episode_data.get_sequence_by_type(elem_type)
            vectors = [encoder.forward(e) for e in sequence]
            return np.stack(vectors)

        vector_seq = np.hstack([encode_and_postprocess(k, v) for k, v in self.items()])
        vector_seq_processed = np.array([self.post_processor.apply(e) for e in vector_seq])
        assert_equal_with_message(vector_seq_processed.ndim, 2, "vector_seq dim")
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
        return sum(encoder.output_size for encoder in self.values())

    @property
    def encode_order(self) -> List[Type[ElementBase]]:
        return list(self.keys())

    def __str__(self) -> str:
        string = "total dim: {}".format(self.dimension)
        for elem_type, encoder in self.items():
            string += "\n{0}: {1}".format(elem_type.__name__, encoder.output_size)
        return string

    @classmethod
    def from_encoders(
        cls, encoder_list: List[EncoderBase], chunk: Optional[MultiEpisodeChunk] = None
    ) -> "EncodingRule":
        rule: EncodingRule = cls()
        for encoder in encoder_list:
            rule[encoder.elem_type] = encoder
        rule.post_processor = IdenticalPostProcessor()

        if chunk is not None:
            # compute normalizer and set to encoder
            vector_seqs = rule.apply_to_multi_episode_chunk(chunk)
            vector_seq_concated = np.concatenate(vector_seqs, axis=0)
            type_dim_table = {t: rule[t].output_size for t in rule.keys()}
            normalizer = ElemCovMatchPostProcessor.from_feature_seqs(
                vector_seq_concated, type_dim_table
            )
            rule.post_processor = normalizer
        return rule
