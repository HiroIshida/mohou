import copy
from dataclasses import dataclass
from typing import List, Optional, Tuple, Type

import numpy as np
import torch
from torch.utils.data import Dataset

from mohou.dataset.sequence_dataset import (
    AutoRegressiveDatasetConfig,
    PaddingSequenceAligner,
    SequenceDataAugmentor,
)
from mohou.encoding_rule import EncodingRule
from mohou.types import EpisodeBundle, ImageBase, RGBImage
from mohou.utils import (
    assert_equal_with_message,
    assert_seq_list_list_compatible,
    flatten_lists,
)


@dataclass
class ChimeraDataset(Dataset):
    image_type: Type[ImageBase]
    image_seqs: List[List[ImageBase]]
    vector_seqs: List[np.ndarray]

    def __post_init__(self):
        assert_equal_with_message(len(self.image_seqs), len(self.vector_seqs), "length of seq")

    def __len__(self) -> int:
        return len(self.image_seqs)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        image_seq = self.image_seqs[idx]
        image_seq_tensor = torch.stack([img.to_tensor() for img in image_seq], dim=0)

        vector_seq = self.vector_seqs[idx]
        vector_seq_tensor = torch.from_numpy(vector_seq).float()
        return image_seq_tensor, vector_seq_tensor

    @classmethod
    def from_bundle(
        cls,
        bundle: EpisodeBundle,
        encoding_rule: EncodingRule,
        dataset_config: Optional[AutoRegressiveDatasetConfig] = None,
    ) -> "ChimeraDataset":

        if dataset_config is None:
            dataset_config = AutoRegressiveDatasetConfig()

        # assume that encoding_rule does not have encoder for image
        for elem_type in encoding_rule.keys():
            assert not issubclass(elem_type, ImageBase)
        vector_seqs: List[np.ndarray] = encoding_rule.apply_to_episode_bundle(bundle)

        # TODO: currently assumes only RGB
        image_seqs: List[List[RGBImage]] = []
        for episode_data in bundle:
            tmp = episode_data.get_sequence_by_type(RGBImage)
            image_seqs.append(tmp.elem_list)

        # data augmentation
        augmentor = SequenceDataAugmentor.from_seqs(vector_seqs, dataset_config)
        vector_seqs_auged = flatten_lists([augmentor.apply(seq) for seq in vector_seqs])
        image_seqs_auged: List[List[RGBImage]] = flatten_lists(
            [[copy.deepcopy(seq) for _ in range(dataset_config.n_aug + 1)] for seq in image_seqs]
        )

        # align seq list
        n_after_termination = dataset_config.n_dummy_after_termination
        assert_seq_list_list_compatible([image_seqs_auged, vector_seqs_auged])
        aligner = PaddingSequenceAligner.from_seqs(image_seqs_auged, n_after_termination)
        image_seqs_aligned = [aligner.apply(seq) for seq in image_seqs_auged]
        vector_seqs_aligned = [aligner.apply(seq) for seq in vector_seqs_auged]
        return cls(RGBImage, image_seqs_aligned, vector_seqs_aligned)  # type: ignore
