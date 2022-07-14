import copy
from dataclasses import dataclass
from typing import Generic, List, Tuple, Type, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from mohou.dataset.sequence_dataset import (
    PaddingSequenceAligner,
    SequenceDataAugmentor,
    SequenceDatasetConfig,
)
from mohou.encoder import ImageEncoder
from mohou.encoding_rule import EncodingRule
from mohou.model import LSTM, AutoEncoderConfig, LSTMConfig
from mohou.model.autoencoder import AutoEncoder
from mohou.model.common import LossDict, ModelBase, ModelConfigBase
from mohou.types import EpisodeBundle, ImageBase, ImageT
from mohou.utils import (
    assert_equal_with_message,
    assert_seq_list_list_compatible,
    flatten_lists,
)


@dataclass
class ChimeraConfig(ModelConfigBase):
    lstm_config: LSTMConfig
    ae_config: Union[AutoEncoderConfig, AutoEncoder]  # TODO(HiroIshida): bit dirty


class Chimera(ModelBase[ChimeraConfig], Generic[ImageT]):
    """Chimera model with lstm and autoencoder
    This is experimental model and the interface will probably be changed later.
    """

    image_type: Type[ImageT]
    lstm: LSTM
    ae: AutoEncoder[ImageT]

    def _setup_from_config(self, config: ChimeraConfig) -> None:
        # TODO(HiroIshida) currently fixed to auto encoder
        self.lstm = LSTM(config.lstm_config)
        if isinstance(config.ae_config, AutoEncoderConfig):
            self.ae = AutoEncoder(config.ae_config)
        elif isinstance(config.ae_config, AutoEncoder):
            ae = config.ae_config
            ae.device = self.lstm.device
            ae.put_on_device()
            self.ae = ae
        else:
            assert False

        self.image_type = self.ae.image_type

    def get_encoder(self) -> ImageEncoder[ImageT]:
        return self.ae.get_encoder()

    def loss(self, sample: Tuple[torch.Tensor, torch.Tensor]) -> LossDict:
        # TODO(HiroIshida) consider weight later
        image_seqs, vector_seqs = sample
        assert image_seqs.ndim == 5
        assert vector_seqs.ndim == 3
        assert image_seqs.shape[0] == vector_seqs.shape[0]  # batch size
        assert image_seqs.shape[1] == vector_seqs.shape[1]  # sequence length

        n_batch, n_seqlen = image_seqs.shape[0], image_seqs.shape[1]

        # for efficiency we encode the image at once
        images_at_once = image_seqs.reshape((n_batch * n_seqlen, *image_seqs.shape[2:]))
        image_features_at_once = self.ae.get_encoder_module()(images_at_once)
        image_feature_seqs = image_features_at_once.reshape(n_batch, n_seqlen, -1)

        # TODO(HiroIshida) tmporary assume default encoding rule order (i.e. image first)
        feature_seqs = torch.concat((image_feature_seqs, vector_seqs), dim=2)

        # compute lstm loss
        feature_seq_input, feature_seq_output_gt = feature_seqs[:, :-1], feature_seqs[:, 1:]
        assert self.config.lstm_config.n_static_context == 0
        static_context = torch.empty(n_batch, 0).to(self.device)
        feature_seq_output, _ = self.lstm.forward(feature_seq_input, static_context)
        pred_loss = torch.mean((feature_seq_output - feature_seq_output_gt) ** 2)

        # compute reconstruction loss
        image_reconst_at_once = self.ae.get_decoder_module()(image_features_at_once)
        reconst_loss = nn.MSELoss()(images_at_once, image_reconst_at_once)

        return LossDict({"prediction": pred_loss, "reconstruction": reconst_loss})


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
    def from_bundle(cls, bundle: EpisodeBundle, encoding_rule: EncodingRule) -> "ChimeraDataset":

        first_elem_type = encoding_rule.encode_order[0]
        assert issubclass(
            first_elem_type, ImageBase
        )  # TODO(HiroIshida) relax this. see the model loss func
        image_type: Type[ImageBase] = first_elem_type
        encoding_rule.delete(image_type)  # because image encoding is done in the chimera model
        vector_seqs: List[np.ndarray] = encoding_rule.apply_to_episode_bundle(bundle)

        image_seqs: List[List[ImageBase]] = []
        for episode_data in bundle:
            tmp = episode_data.get_sequence_by_type(image_type)
            image_seqs.append(tmp.elem_list)

        # data augmentation
        config = SequenceDatasetConfig()
        augmentor = SequenceDataAugmentor.from_seqs(vector_seqs, config)
        vector_seqs_auged = flatten_lists([augmentor.apply(seq) for seq in vector_seqs])
        image_seqs_auged: List[List[ImageBase]] = flatten_lists(
            [[copy.deepcopy(seq) for _ in range(config.n_aug + 1)] for seq in image_seqs]
        )

        for image_seq in image_seqs_auged:
            for i in range(len(image_seq)):
                image_seq[i] = image_seq[i].randomize()

        # align seq list
        n_after_termination = 20  # TODO(HiroIshida) from config
        assert_seq_list_list_compatible([image_seqs_auged, vector_seqs_auged])
        aligner = PaddingSequenceAligner.from_seqs(image_seqs_auged, n_after_termination)
        image_seqs_aligned = [aligner.apply(seq) for seq in image_seqs_auged]
        vector_seqs_aligned = [aligner.apply(seq) for seq in vector_seqs_auged]
        return cls(image_type, image_seqs_aligned, vector_seqs_aligned)  # type: ignore
