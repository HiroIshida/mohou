import copy
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Type

import numpy as np

from mohou.encoder import EncoderBase
from mohou.types import (
    CompositeImageBase,
    ElementBase,
    ElementDict,
    EpisodeBundle,
    EpisodeData,
    PrimitiveElementBase,
)
from mohou.utils import (
    DataclassLightMixin,
    abstract_attribute,
    assert_equal_with_message,
    get_bound_list,
)

logger = logging.getLogger(__name__)


class LocalBalancer(ABC):
    bound: slice = abstract_attribute()

    @property
    def dim(self) -> int:
        return self.bound.stop - self.bound.start

    @abstractmethod
    def apply(self, vec: np.ndarray) -> None:
        pass

    @abstractmethod
    def inverse_apply(self, vec: np.ndarray) -> None:
        pass


class NullLocalBalancer(LocalBalancer, DataclassLightMixin):
    bound: slice

    def apply(self, vec: np.ndarray) -> None:
        pass

    def inverse_apply(self, vec: np.ndarray) -> None:
        pass


class ActiveLocalBalancer(LocalBalancer, DataclassLightMixin):
    bound: slice
    mean: np.ndarray
    cov: np.ndarray
    scaled_primary_std: float

    def apply(self, vec: np.ndarray) -> None:
        vec[self.bound] = (vec[self.bound] - self.mean) / self.scaled_primary_std

    def inverse_apply(self, vec: np.ndarray) -> None:
        vec[self.bound] = (vec[self.bound] * self.scaled_primary_std) + self.mean


@dataclass
class CovarianceBalancer:
    type_balancer_table: Dict[Type[ElementBase], LocalBalancer]

    def __post_init__(self):
        self.check_bounds()

    def check_bounds(self) -> None:
        dims = [balancer.dim for balancer in self.type_balancer_table.values()]
        bounds = [balancer.bound for balancer in self.type_balancer_table.values()]
        assert bounds[0].start == 0
        for i in range(len(bounds) - 1):
            assert bounds[i].stop == bounds[i + 1].start
        assert bounds[-1].stop == sum(dims)

    def update(self):
        dims = [balancer.dim for balancer in self.type_balancer_table.values()]

        # update bounds
        bounds_new = get_bound_list(dims)
        for i, balancer in enumerate(self.type_balancer_table.values()):
            balancer.bound = bounds_new[i]
        self.check_bounds()

        # update scaled_primary_std
        active_balancers = [
            b for b in self.type_balancer_table.values() if isinstance(b, ActiveLocalBalancer)
        ]
        covs = [b.cov for b in active_balancers]
        sp_stds = self._compute_scaled_primary_stds(covs)
        for sp_std, balancer in zip(sp_stds, active_balancers):
            balancer.scaled_primary_std = sp_std

    def delete(self, elem_type: Type[ElementBase]) -> None:
        self.type_balancer_table.pop(elem_type)
        self.update()

    def mark_null(self, elem_type: Type[ElementBase]) -> None:
        balancer = self.type_balancer_table[elem_type]
        new_balancer = NullLocalBalancer(balancer.bound)
        self.type_balancer_table[elem_type] = new_balancer
        self.update()

    @staticmethod
    def get_null_only_table(
        type_dim_table: Dict[Type[ElementBase], int]
    ) -> Dict[Type[ElementBase], LocalBalancer]:
        type_balancer_table: Dict[Type[ElementBase], LocalBalancer] = {}
        bounds = get_bound_list(list(type_dim_table.values()))
        for key, bound in zip(type_dim_table.keys(), bounds):
            type_balancer_table[key] = NullLocalBalancer(bound)
        return type_balancer_table

    @classmethod
    def null(cls, type_dim_table: Dict[Type[ElementBase], int]) -> "CovarianceBalancer":
        """create balancer which does nothing (pass through)"""
        return cls(cls.get_null_only_table(type_dim_table))

    @classmethod
    def from_feature_seqs(
        cls,
        feature_seq: np.ndarray,
        type_dim_table: Dict[Type[ElementBase], int],
        type_active_table: Optional[Dict[Type[ElementBase], bool]] = None,
    ) -> "CovarianceBalancer":
        # NOTE: please note that we take advantage of dict's OrderedDict characteritic
        # in this implementation

        if type_active_table is None:
            type_active_table = {}
            for key in type_dim_table.keys():
                type_active_table[key] = True

        # initialize type_balancer_table will all NullLocalBalancer
        # initialization here to preserve table key order
        type_balancer_table = cls.get_null_only_table(type_dim_table)
        bounds = get_bound_list(list(type_dim_table.values()))

        # get active bounds
        active_bounds = []
        for i, key in enumerate(type_dim_table.keys()):
            is_active = type_active_table[key]
            if is_active:
                active_bounds.append(bounds[i])

        # create ActiveLocalBalancer for all active tye
        means, covs = cls._compute_means_and_covs(feature_seq, active_bounds)
        scaled_primary_stds = cls._compute_scaled_primary_stds(covs)
        for key in type_balancer_table.keys():
            is_active = type_active_table[key]
            if is_active:
                bound = active_bounds.pop(0)
                mean = means.pop(0)
                cov = covs.pop(0)
                std = scaled_primary_stds.pop(0)
                lb = ActiveLocalBalancer(bound, mean, cov, std)
                type_balancer_table[key] = lb
        assert len(means) == len(covs) == len(scaled_primary_stds) == len(active_bounds) == 0

        return cls(type_balancer_table)

    @staticmethod
    def _compute_means_and_covs(
        feature_seq: np.ndarray, active_bounds: List[slice]
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        def is_binary_sequence(seq) -> bool:
            return len(set(seq.flatten().tolist())) == 2

        means: List[np.ndarray] = []
        covs: List[np.ndarray] = []
        for bound in active_bounds:
            parital_seq = feature_seq[:, bound]
            dim = parital_seq.shape[1]

            if is_binary_sequence(parital_seq):
                assert (
                    dim == 1
                ), "currently we assume binary seq is only 1dim"  # TODO(HiroIshida): remove
                min_val, max_val = np.min(parital_seq), np.max(parital_seq)
                cov = np.diag(np.ones(dim))
                mean = np.array([0.5 * (min_val + max_val)])
            else:
                mean = np.mean(parital_seq, axis=0)
                cov = np.cov(parital_seq.T)
                if cov.ndim == 0:  # unfortunately, np.cov return 0 dim array instead of 1x1
                    cov = np.expand_dims(cov, axis=0)
                    cov = np.array([[cov.item()]])
            means.append(mean)
            covs.append(cov)
        return means, covs

    @staticmethod
    def _compute_scaled_primary_stds(covs: List[np.ndarray]) -> List[float]:
        primary_stds = []
        for cov in covs:
            eig_values, _ = np.linalg.eig(cov)
            max_eig = max(eig_values)
            primary_stds.append(np.sqrt(max_eig))
        scaled_primary_stds = [std / max(primary_stds) for std in primary_stds]
        return scaled_primary_stds

    def apply(self, vec: np.ndarray) -> np.ndarray:
        vec_out = copy.deepcopy(vec)
        for lb in self.type_balancer_table.values():
            lb.apply(vec_out)
        return vec_out

    def inverse_apply(self, vec: np.ndarray) -> np.ndarray:
        vec_out = copy.deepcopy(vec)
        for lb in self.type_balancer_table.values():
            lb.inverse_apply(vec_out)
        return vec_out


class EncodingRule(Dict[Type[ElementBase], EncoderBase]):
    covariance_balancer: CovarianceBalancer

    def pop(self, *args):
        # As we have delete function, it is bit confusing
        raise NotImplementedError  # delete this method if Dict

    def delete(self, elem_type: Type[ElementBase]):
        super().pop(elem_type)
        self.covariance_balancer.delete(elem_type)

    def apply(self, elem_dict: ElementDict) -> np.ndarray:
        vector_list = []
        for elem_type, encoder in self.items():
            vector = encoder.forward(elem_dict[elem_type])
            vector_list.append(vector)
        return self.covariance_balancer.apply(np.hstack(vector_list))

    def inverse_apply(self, vector_processed: np.ndarray) -> ElementDict:
        def split_vector(vector: np.ndarray, size_list: List[int]):
            head = 0
            vector_list = []
            for i, size in enumerate(size_list):
                tail = head + size
                vector_list.append(vector[head:tail])
                head = tail
            return vector_list

        vector = self.covariance_balancer.inverse_apply(vector_processed)
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
        vector_seq_processed = np.array([self.covariance_balancer.apply(e) for e in vector_seq])
        assert_equal_with_message(vector_seq_processed.ndim, 2, "vector_seq dim")
        return vector_seq_processed

    def apply_to_episode_bundle(self, bundle: EpisodeBundle) -> List[np.ndarray]:
        def elem_types_to_primitive_elem_set(elem_type_list: List[Type[ElementBase]]):
            primitve_elem_type_list = []
            for elem_type in elem_type_list:
                if issubclass(elem_type, PrimitiveElementBase):
                    primitve_elem_type_list.append(elem_type)
                elif issubclass(elem_type, CompositeImageBase):
                    primitve_elem_type_list.extend(elem_type.image_types)
            return set(primitve_elem_type_list)

        bundle_elem_types = elem_types_to_primitive_elem_set(list(bundle.types()))
        required_elem_types = elem_types_to_primitive_elem_set(list(self.keys()))
        assert required_elem_types <= bundle_elem_types

        vector_seq_list = [self.apply_to_episode_data(data) for data in bundle]

        assert vector_seq_list[0].ndim == 2
        return vector_seq_list

    @property
    def dimension(self) -> int:
        return sum(encoder.output_size for encoder in self.values())

    @property
    def encode_order(self) -> List[Type[ElementBase]]:
        return list(self.keys())

    @property
    def type_bound_table(self) -> Dict[Type[ElementBase], slice]:
        dims = [encoder.output_size for encoder in self.values()]
        bounds = get_bound_list(dims)

        table = {}
        for encoder, bound in zip(self.values(), bounds):
            table[encoder.elem_type] = bound
        return table

    def __str__(self) -> str:
        string = "total dim: {}".format(self.dimension)
        for elem_type, encoder in self.items():
            string += "\n{0}: {1}".format(elem_type.__name__, encoder.output_size)
        return string

    @classmethod
    def from_encoders(
        cls, encoder_list: List[EncoderBase], bundle: Optional[EpisodeBundle] = None
    ) -> "EncodingRule":
        rule: EncodingRule = cls()
        for encoder in encoder_list:
            rule[encoder.elem_type] = encoder

        type_dim_table = {t: rule[t].output_size for t in rule.keys()}
        rule.covariance_balancer = CovarianceBalancer.null(type_dim_table)

        if bundle is not None:
            # compute normalizer and set to encoder
            vector_seqs = rule.apply_to_episode_bundle(bundle)
            vector_seq_concated = np.concatenate(vector_seqs, axis=0)
            normalizer = CovarianceBalancer.from_feature_seqs(vector_seq_concated, type_dim_table)
            rule.covariance_balancer = normalizer
        return rule
