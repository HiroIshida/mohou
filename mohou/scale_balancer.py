from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List

import numpy as np

from mohou.utils import assert_equal_with_message


class ScaleBalancerBase(ABC):
    @abstractmethod
    def apply(self, vec: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def inverse_apply(self, vec: np.ndarray) -> np.ndarray:
        pass


class NullScaleBalancer(ScaleBalancerBase):
    def apply(self, vec: np.ndarray) -> np.ndarray:
        return vec

    def inverse_apply(self, vec: np.ndarray) -> np.ndarray:
        return vec


@dataclass
class ScaleBalancer(ScaleBalancerBase):
    means: np.ndarray
    widths: np.ndarray

    def __post_init__(self):
        # field validation
        assert self.means.ndim == 1
        assert self.widths.ndim == 1
        assert len(self.means) == len(self.widths)
        assert np.min(self.widths) > 1e-10

    @property
    def dimension(self) -> int:
        return len(self.means)

    @classmethod
    def from_array_list(cls, feature_vectors_list: List[np.ndarray]) -> "ScaleBalancer":
        """
        Args:
            feature_vectors_list: list of feture_vector sequence created by application of
                encoding rule.
        """

        # data validatoin
        n_dof_ref = feature_vectors_list[0].shape[1]
        for vectors in feature_vectors_list:
            vectors.ndim == 2
            n_seqlen, n_dof = vectors.shape
            assert_equal_with_message(n_dof, n_dof_ref, "n_dof")

        means_per_episode_list = []
        widths_per_episode_list = []
        for vectors in feature_vectors_list:
            means_per_episode = np.mean(vectors, axis=0)
            means_per_episode_list.append(means_per_episode)

            mins_per_episode = np.min(vectors, axis=0)
            maxs_per_episode = np.max(vectors, axis=0)
            widths_per_episode = maxs_per_episode - mins_per_episode
            widths_per_episode_list.append(widths_per_episode)

        # average over all episodes
        n_episode = len(feature_vectors_list)
        average_means: np.ndarray = sum(means_per_episode_list) / n_episode
        average_widths: np.ndarray = sum(widths_per_episode_list) / n_episode
        return cls(average_means, average_widths)

    def apply(self, vec: np.ndarray) -> np.ndarray:
        return (vec - self.means) / self.widths

    def inverse_apply(self, vec: np.ndarray) -> np.ndarray:
        return (vec * self.widths) + self.means
