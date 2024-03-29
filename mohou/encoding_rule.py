import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from typing import (
    Dict,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
)

import numpy as np
import torch

from mohou.encoder import EncoderBase, HasAModel, ImageEncoder, VectorIdenticalEncoder
from mohou.types import (
    AngleVector,
    AnotherGripperState,
    CompositeImageBase,
    DepthImage,
    ElementBase,
    ElementDict,
    EpisodeBundle,
    EpisodeData,
    GripperState,
    PrimitiveElementBase,
    RGBDImage,
    RGBImage,
    TerminateFlag,
)
from mohou.utils import assert_equal_with_message, get_bound_list

logger = logging.getLogger(__name__)


ScaleBalancerT = TypeVar("ScaleBalancerT", bound="ScaleBalancerBase")
ArrayT = TypeVar("ArrayT", bound=Union[torch.Tensor, np.ndarray])


class ScaleBalancerBase(ABC):
    @classmethod
    def get_json_file_path(cls, project_path: Path, create_dir: bool = False) -> Path:
        """get the json file path that will be used for dumping and loading
        Note that dumping / loading the scale balancer is still an experimental feature.
        """
        save_dir_path = project_path / "experimental"
        if create_dir:
            save_dir_path.mkdir(exist_ok=True)
        file_name = cls.__name__ + ".json"
        json_file_path = save_dir_path / file_name
        return json_file_path

    @abstractmethod
    def apply(self, arr: ArrayT) -> ArrayT:
        pass

    @abstractmethod
    def inverse_apply(self, arr: ArrayT) -> ArrayT:
        pass

    @abstractmethod
    def dump(self, project_path: Path) -> None:
        pass

    @classmethod
    @abstractmethod
    def load(cls: Type[ScaleBalancerT], project_path: Path) -> ScaleBalancerT:
        pass


@dataclass(frozen=True)
class IdenticalScaleBalancer(ScaleBalancerBase):
    def apply(self, arr: ArrayT) -> ArrayT:
        return arr

    def inverse_apply(self, arr: ArrayT) -> ArrayT:
        return arr

    def dump(self, project_path: Path) -> None:
        json_file_path = self.get_json_file_path(project_path, create_dir=True)
        with json_file_path.open(mode="w") as f:
            json.dump({}, f)

    @classmethod
    def load(cls, project_path: Path) -> "IdenticalScaleBalancer":
        logger.warning("NOTE: This feature may be dropped without any notification")
        json_file_path = cls.get_json_file_path(project_path)
        assert json_file_path.exists()
        with json_file_path.open(mode="r") as f:
            kwargs = json.load(f)
        # because actually this class does not have any members
        assert len(kwargs) == 0
        return cls()

    def __eq__(self, other: object) -> bool:
        keys = self.__dataclass_fields__.keys()  # type: ignore # mypy's bag
        assert set(keys) == set()
        return isinstance(other, IdenticalScaleBalancer)


@dataclass(frozen=True)
class CovarianceBasedScaleBalancer(ScaleBalancerBase):
    dims: List[int]
    means: List[np.ndarray]
    scaled_stds: List[float]

    def __eq__(self, other: object) -> bool:
        keys = self.__dataclass_fields__.keys()  # type: ignore # mypy's bag
        assert set(keys) == {"dims", "means", "scaled_stds"}

        if not isinstance(other, CovarianceBasedScaleBalancer):
            return NotImplemented
        assert type(self) is type(other)

        if self.dims != other.dims:
            return False
        if not np.allclose(np.hstack(self.means), np.hstack(other.means), atol=1e-6):
            return False
        if self.scaled_stds != other.scaled_stds:
            return False

        return True

    def dump(self, project_path: Path):
        json_file_path = self.get_json_file_path(project_path, create_dir=True)
        dic = {
            "dims": self.dims,
            "means": [mean.tolist() for mean in self.means],
            "scaled_stds": self.scaled_stds,
        }
        with json_file_path.open(mode="w") as f:
            json.dump(dic, f)

    @classmethod
    def load(cls, project_path: Path) -> "CovarianceBasedScaleBalancer":
        logger.warning("NOTE: This feature may be dropped without any notification")
        json_file_path = cls.get_json_file_path(project_path)
        assert json_file_path.exists()
        with json_file_path.open(mode="r") as f:
            kwargs = json.load(f)
        means = kwargs["means"]
        kwargs["means"] = [np.array(mean) for mean in means]
        return cls(**kwargs)

    def __post_init__(self):
        for i, dim in enumerate(self.dims):
            assert_equal_with_message(len(self.means), len(self.dims), "len")
            assert_equal_with_message(len(self.scaled_stds), len(self.dims), "len")
            assert_equal_with_message(self.means[i].shape, (dim,), "mean shape of {}".format(i))

    @staticmethod
    def get_bound_list(dims: List[int]) -> List[slice]:
        bound_list = []
        head = 0
        for dim in dims:
            bound_list.append(slice(head, head + dim))
            head += dim
        return bound_list

    @staticmethod
    def is_binary_sequence(partial_feature_seq: np.ndarray):
        return len(set(partial_feature_seq.flatten().tolist())) == 2

    @classmethod
    def from_feature_seqs(cls, feature_seq: np.ndarray, dims: List[int]):
        assert_equal_with_message(feature_seq.ndim, 2, "feature_seq.ndim")
        means = []
        max_stds = []
        for rang in cls.get_bound_list(dims):
            feature_seq_partial = feature_seq[:, rang]
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

            eig_values, _ = np.linalg.eig(cov)
            max_std = np.sqrt(max(eig_values))

            # Note that np.cov(pts.T) outputs some non 0 value even if the input points
            # are all at a single point (degenerated). So, max_eig > 0.0 cannot check
            # the data degeneration. Thus...
            assert max_std > 1e-7

            means.append(mean)
            max_stds.append(max_std)

        scaled_stds = list(np.array(max_stds) / max(max_stds))
        return cls(dims, means, scaled_stds)

    def _apply(self, arr: ArrayT, inverse: bool) -> ArrayT:

        assert arr.ndim in [1, 2]

        dim = len(arr) if arr.ndim == 1 else arr.shape[1]
        assert_equal_with_message(dim, sum(self.dims), "vector total dim")

        mean = np.hstack(self.means)
        std_list = [np.ones(dim) * std for dim, std, in zip(self.dims, self.scaled_stds)]
        std = np.hstack(std_list)

        if isinstance(arr, torch.Tensor):
            mean = torch.from_numpy(mean).float().to(arr.device)  # type: ignore
            std = torch.from_numpy(std).float().to(arr.device)  # type: ignore

        if inverse:
            return (arr * std) + mean  # type: ignore
        else:
            return (arr - mean) / std  # type: ignore

    def apply(self, arr: ArrayT) -> ArrayT:
        return self._apply(arr, False)

    def inverse_apply(self, arr: ArrayT) -> ArrayT:
        return self._apply(arr, True)


class EncodingRuleBase(Mapping[Type[ElementBase], EncoderBase]):
    @abstractmethod
    def apply(self, elem_dict: ElementDict) -> np.ndarray:
        pass

    @abstractmethod
    def inverse_apply(self, vector_processed: np.ndarray) -> ElementDict:
        pass

    @property
    @abstractmethod
    def dimension_list(self) -> List[int]:
        """get dimension list for each encoders output"""

    @property
    def type_bound_table(self) -> Dict[Type[ElementBase], slice]:
        bounds = get_bound_list(self.dimension_list)

        table = {}
        for encoder, bound in zip(self.values(), bounds):
            table[encoder.elem_type] = bound
        return table

    def apply_to_episode_data(self, episode_data: EpisodeData) -> np.ndarray:
        vec_list = [self.apply(episode_data[i]) for i in range(len(episode_data))]
        return np.array(vec_list)

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

    def get_device(self) -> Optional[torch.device]:
        device_list = []
        for encoder in self.values():
            if isinstance(encoder, HasAModel):
                device_list.append(encoder.get_device())
        if len(device_list) == 0:
            return None
        device_set = set(device_list)
        assert len(device_set) == 1, "Do not mix more than 1 devices"
        return device_set.pop()

    def set_device(self, device: torch.device) -> None:
        for encoder in self.values():
            if isinstance(encoder, HasAModel):
                encoder.set_device(device)


_default_encoding_rule_cache: Dict[Tuple, "EncodingRule"] = {}


class EncodingRule(Dict[Type[ElementBase], EncoderBase], EncodingRuleBase):
    scale_balancer: ScaleBalancerBase

    def apply(self, elem_dict: ElementDict) -> np.ndarray:
        vector_list = []
        for elem_type, encoder in self.items():
            vector = encoder.forward(elem_dict[elem_type])
            vector_list.append(vector)
        return self.scale_balancer.apply(np.hstack(vector_list))

    def inverse_apply(self, vector_processed: np.ndarray) -> ElementDict:
        def split_vector(vector: np.ndarray, size_list: List[int]):
            head = 0
            vector_list = []
            for i, size in enumerate(size_list):
                tail = head + size
                vector_list.append(vector[head:tail])
                head = tail
            return vector_list

        vector = self.scale_balancer.inverse_apply(vector_processed)
        size_list = [encoder.output_size for elem_type, encoder in self.items()]
        vector_list = split_vector(vector, size_list)

        elem_dict = ElementDict([])
        for vec, (elem_type, encoder) in zip(vector_list, self.items()):
            elem_dict[elem_type] = encoder.backward(vec)
        return elem_dict

    @property
    def dimension(self) -> int:
        return sum(encoder.output_size for encoder in self.values())

    @property
    def dimension_list(self) -> List[int]:
        return [encoder.output_size for encoder in self.values()]

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
        cls,
        encoder_list: Sequence[EncoderBase],
        bundle: Optional[EpisodeBundle] = None,
        scale_balancer: Optional[ScaleBalancerBase] = None,
    ) -> "EncodingRule":
        """Create EncodingRule from encoder_list
        Args:
            encoder_list: list of encoder. Order of the list is important and preserved.
            bundle: If set, ScaleBalancer will created using bundle
            scale_balancer: use this scale balancer if set.

        bundle != None and scale_balancer != None is never accepted
        """
        # NOTE: currently we can load cached balancer. But loading cached entire encoder is,
        # of course the future direction. However, the difficulty mainly lies in the serializing
        # ImageEncoder which contains lambda functions.

        rule: EncodingRule = cls()
        for encoder in encoder_list:
            rule[encoder.elem_type] = encoder
        rule.scale_balancer = IdenticalScaleBalancer()  # tmp

        if scale_balancer is not None:
            assert not bundle
            rule.scale_balancer = scale_balancer

        if bundle is not None:
            assert not scale_balancer
            dims = [encoder.output_size for encoder in rule.values()]
            # compute normalizer and set to encoder
            vector_seqs = rule.apply_to_episode_bundle(bundle)
            vector_seq_concated = np.concatenate(vector_seqs, axis=0)
            rule.scale_balancer = CovarianceBasedScaleBalancer.from_feature_seqs(
                vector_seq_concated, dims
            )
        return rule

    @classmethod
    def create_default(
        cls,
        project_path: Path,
        include_image_encoder: bool = True,
        use_balancer: bool = True,
    ) -> "EncodingRule":

        cache_key = (project_path, include_image_encoder, use_balancer)
        if cache_key in _default_encoding_rule_cache:
            return _default_encoding_rule_cache[cache_key]

        bundle = EpisodeBundle.load(project_path)
        bundle_spec = bundle.spec

        encoders: List[EncoderBase] = []

        if include_image_encoder:
            image_encoder = ImageEncoder.create_default(project_path)
            encoders.append(image_encoder)

        if AngleVector in bundle_spec.type_shape_table:
            av_dim = bundle_spec.type_shape_table[AngleVector][0]
            av_idendical_encoder = VectorIdenticalEncoder.create(AngleVector, av_dim)
            encoders.append(av_idendical_encoder)

        if GripperState in bundle_spec.type_shape_table:
            gs_identital_func = VectorIdenticalEncoder.create(
                GripperState, bundle_spec.type_shape_table[GripperState][0]
            )
            encoders.append(gs_identital_func)

        if AnotherGripperState in bundle_spec.type_shape_table:
            ags_identital_func = VectorIdenticalEncoder.create(
                AnotherGripperState, bundle_spec.type_shape_table[AnotherGripperState][0]
            )
            encoders.append(ags_identital_func)

        tf_identical_func = VectorIdenticalEncoder.create(TerminateFlag, 1)
        encoders.append(tf_identical_func)

        p = CovarianceBasedScaleBalancer.get_json_file_path(project_path)
        if p.exists():  # use cached balacner
            balancer: Optional[CovarianceBasedScaleBalancer]
            if use_balancer:
                logger.warning(
                    "warn: loading cached CovarianceBasedScaleBalancer. This feature is experimental."
                )
                balancer = CovarianceBasedScaleBalancer.load(project_path)
            else:
                balancer = None
            encoding_rule = EncodingRule.from_encoders(
                encoders, bundle=None, scale_balancer=balancer
            )
        else:
            bundle_for_balancer: Optional[EpisodeBundle]
            if use_balancer:
                bundle_for_balancer = bundle
            else:
                bundle_for_balancer = None
            encoding_rule = EncodingRule.from_encoders(
                encoders, bundle=bundle_for_balancer, scale_balancer=None
            )

        # TODO: Move The following check to unittest? but it's diffcult becaues
        # using this function pre-require the existence of trained AE ...
        # so temporary, check using assertion

        # check if ordered propery. Keeping this order is important to satisfy the
        # backward-compatibility of this function.
        order_definition: List[Type[ElementBase]] = [
            RGBImage,
            DepthImage,
            RGBDImage,
            AngleVector,
            GripperState,
            AnotherGripperState,
            TerminateFlag,
        ]
        indices = [order_definition.index(key) for key in encoding_rule.keys()]
        is_ordered_propery = sorted(indices) == indices
        assert is_ordered_propery

        # update cache
        _default_encoding_rule_cache[cache_key] = encoding_rule

        return encoding_rule


@dataclass
class CompositeEncodingRule(EncodingRuleBase):
    rules: List[EncodingRule]

    def __post_init__(self):
        key_set_list = [set(rule.keys()) for rule in self.rules]
        set_entire = set.union(*key_set_list)
        no_intersection = len(set_entire) == sum([len(rule.keys()) for rule in self.rules])
        assert no_intersection

    def apply(self, edict: ElementDict) -> np.ndarray:
        vec = np.hstack([rule.apply(edict) for rule in self.rules])
        return vec

    def inverse_apply(self, vector: np.ndarray) -> ElementDict:
        head = 0
        elems = []
        for rule in self.rules:
            tail = head + rule.dimension
            edict = rule.inverse_apply(vector[head:tail])
            elems.extend(list(edict.values()))
            head = tail
        edict_merged = ElementDict(elems)
        return edict_merged

    @property
    def dimension_list(self) -> List[int]:
        dims = []
        for rule in self.rules:
            for encoder in rule.values():
                dims.append(encoder.output_size)
        return dims

    def __getitem__(self, key: Type[ElementBase]) -> EncoderBase:
        for rule in self.rules:
            if key in rule:
                return rule[key]
        raise KeyError

    def __iter__(self) -> Iterator[Type[ElementBase]]:
        return chain(*[rule.__iter__() for rule in self.rules])

    def __len__(self) -> int:
        return sum([len(rule) for rule in self.rules])
