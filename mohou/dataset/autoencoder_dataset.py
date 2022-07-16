import copy
import logging
from dataclasses import dataclass
from typing import Generic, List, Optional, Type

import torch
from torch.utils.data import Dataset

from mohou.types import EpisodeBundle, ImageT

logger = logging.getLogger(__name__)


@dataclass
class AutoEncoderDatasetConfig:
    batch_augment_factor: int = 2  # if you have large enough RAM, set to large (like 4)

    def __post_init__(self):
        assert self.batch_augment_factor >= 0
        logger.info("autoencoder dataset config: {}".format(self))


@dataclass
class AutoEncoderDataset(Dataset, Generic[ImageT]):
    image_type: Type[ImageT]
    image_list: List[ImageT]

    def __len__(self) -> int:
        return len(self.image_list)

    def __getitem__(self, idx) -> torch.Tensor:
        return self.image_list[idx].to_tensor()

    @classmethod
    def from_bundle(
        cls,
        bundle: EpisodeBundle,
        image_type: Type[ImageT],
        augconfig: Optional[AutoEncoderDatasetConfig] = None,
    ) -> "AutoEncoderDataset":

        if augconfig is None:
            augconfig = AutoEncoderDatasetConfig()

        image_list: List[ImageT] = []
        for episode_data in bundle:
            image_list.extend(episode_data.get_sequence_by_type(image_type))

        image_list_rand = copy.deepcopy(image_list)
        for i in range(augconfig.batch_augment_factor):
            image_list_rand.extend([copy.deepcopy(image).randomize() for image in image_list])

        return cls(image_type, image_list_rand)
