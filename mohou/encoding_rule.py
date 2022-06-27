import copy
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Type

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
from mohou.utils import assert_equal_with_message, get_bound_list

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


class LocalProcessor(ABC):
    @abstractmethod
    def apply_inplace(self, vec: np.ndarray) -> None:
        pass

    @abstractmethod
    def inverse_apply_inplace(self, vec: np.ndarray) -> None:
        pass


@dataclass
class PassThroughLocalProcessor(LocalProcessor):
    def apply_inplace(self, vec: np.ndarray) -> None:
        return

    def inverse_apply_inplace(self, vec: np.ndarray) -> None:
        return


@dataclass
class ScalingLocalProcessor(LocalProcessor):
    elem_type: Type[ElementBase]
    bound: slice
    mean: np.ndarray
    cov: np.ndarray
    scaled_primary_std: Optional[float] = None  # take float except before initialization

    def __post_init__(self) -> None:
        dim = len(self.mean)
        assert_equal_with_message(
            self.mean.shape, (dim,), "mean shape of {}".format(self.elem_type)
        )
        assert_equal_with_message(
            self.cov.shape, (dim, dim), "cov shape of {}".format(self.elem_type)
        )

    def apply_inplace(self, vec: np.ndarray) -> None:
        assert self.scaled_primary_std is not None
        vec_new = (vec[self.bound] - self.mean) / self.scaled_primary_std  # type: ignore
        vec[self.bound] = vec_new

    def inverse_apply_inplace(self, vec: np.ndarray) -> None:
        vec_new = (vec[self.bound] * self.scaled_primary_std) + self.mean
        vec[self.bound] = vec_new


@dataclass
class ElemCovMatchPostProcessor(PostProcessor):
    type_dim_table: Dict[Type[ElementBase], int]
    type_local_proc_table: Dict[Type[ElementBase], LocalProcessor]

    def __post_init__(self) -> None:
        self.udpate()

    def delete(self, elem_type: Type[ElementBase]) -> None:
        self.type_dim_table.pop(elem_type)
        self.type_local_proc_table.pop(elem_type)
        self.udpate()

    @property
    def dimension(self) -> int:
        return sum(self.type_dim_table.values())

    @property
    def active_local_processors(self) -> List[ScalingLocalProcessor]:
        active_local_proc_list: List[ScalingLocalProcessor] = []
        for local_proc in self.type_local_proc_table.values():
            if isinstance(local_proc, ScalingLocalProcessor):
                active_local_proc_list.append(local_proc)
        return active_local_proc_list

    def udpate(self) -> None:
        self._update_primal_stds()
        self._update_bounds()

    def _update_primal_stds(self):
        def get_max_std(cov) -> float:
            eig_values, _ = np.linalg.eig(cov)
            max_eig_cov = max(eig_values)
            return np.sqrt(max_eig_cov)

        n_active = len(self.active_local_processors)

        primal_std_list = [
            get_max_std(local_proc.cov) for local_proc in self.active_local_processors
        ]
        max_primal_std = max(primal_std_list)
        scaled_pirmal_std_list = [std / max_primal_std for std in primal_std_list]

        # assign
        for i in range(n_active):
            self.active_local_processors[i].scaled_primary_std = scaled_pirmal_std_list[i]

    def _update_bounds(self):
        dims = list(self.type_dim_table.values())
        for local_proc, bound in zip(self.active_local_processors, get_bound_list(dims)):
            local_proc.bound = bound

    @staticmethod
    def is_binary_sequence(partial_feature_seq: np.ndarray):
        return len(set(partial_feature_seq.flatten().tolist())) == 2

    @classmethod
    def from_feature_seqs(
        cls, feature_seq: np.ndarray, type_dim_table: Dict[Type[ElementBase], int]
    ):
        assert_equal_with_message(feature_seq.ndim, 2, "feature_seq.ndim")
        dims = list(type_dim_table.values())

        type_local_proc_table: Dict[Type[ElementBase], LocalProcessor] = {}

        for elem_type, bound in zip(type_dim_table.keys(), get_bound_list(dims)):
            feature_seq_partial = feature_seq[:, bound]
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

            type_local_proc_table[elem_type] = ScalingLocalProcessor(elem_type, bound, mean, cov)
        return cls(type_dim_table, type_local_proc_table)

    def check_input_vector(self, vec: np.ndarray) -> None:
        assert_equal_with_message(vec.ndim, 1, "vector dim")
        assert_equal_with_message(len(vec), self.dimension, "vector total dim")

    def apply(self, vec: np.ndarray) -> np.ndarray:
        self.check_input_vector(vec)
        self.udpate()

        vec_out = copy.deepcopy(vec)
        for local_proc in self.type_local_proc_table.values():
            local_proc.apply_inplace(vec_out)
        return vec_out

    def inverse_apply(self, vec: np.ndarray) -> np.ndarray:
        self.check_input_vector(vec)
        self.udpate()

        vec_out = copy.deepcopy(vec)
        for local_proc in self.type_local_proc_table.values():
            local_proc.inverse_apply_inplace(vec_out)
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
