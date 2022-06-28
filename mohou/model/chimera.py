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
from mohou.types import ImageBase, ImageT, MultiEpisodeChunk


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
        """compute loss
        sample: tuple of (image_seqs, vector_seqs_seq)
        each image_seqs corresponds to a element of vector_seqs_seq (e.g. vector_seqs_seq[0])
        in the sense that both are originated from the same episode.
        In contrast to image_seqs, vector_seqs is augmented by trajectory augmentation.
        """

        # TODO(HiroIshida) take weight as input
        image_seqs, vector_seqs_seq = sample
        assert image_seqs.ndim == 5
        assert vector_seqs_seq.ndim == 4

        n_batch, n_aug, n_seqlen, n_vecdim = vector_seqs_seq.shape
        assert image_seqs.shape[0] == n_batch
        assert image_seqs.shape[1] == n_seqlen

        # for efficiency we encode the image at once
        images_at_once = image_seqs.reshape((n_batch * n_seqlen, *image_seqs.shape[2:]))
        image_features_at_once = self.ae.get_encoder_module()(images_at_once)

        # compute reconstruction loss
        image_reconst_at_once = self.ae.get_decoder_module()(image_features_at_once)
        reconst_loss = nn.MSELoss()(images_at_once, image_reconst_at_once)

        # compute prediction loss
        # note that prediction loss is done for all augmentd sequence and finally take a mean
        pred_loss_list = []
        image_feature_seqs = image_features_at_once.reshape(n_batch, n_seqlen, -1)

        # NOTE: swapaxes make thing easier, because now each vector_seqs correponds to image_seqs
        for vector_seqs in vector_seqs_seq.swapaxes(0, 1):
            # TODO(HiroIshida) tmporary assume default encoding rule order (i.e. image first)
            feature_seqs = torch.concat((image_feature_seqs, vector_seqs), dim=2)

            feature_seq_input, feature_seq_output_gt = feature_seqs[:, :-1], feature_seqs[:, 1:]
            assert self.config.lstm_config.n_static_context == 0
            static_context = torch.empty(n_batch, 0).to(self.device)
            feature_seq_output = self.lstm.forward(feature_seq_input, static_context)
            pred_loss = torch.mean((feature_seq_output - feature_seq_output_gt) ** 2)
            pred_loss_list.append(pred_loss)
        pred_loss_mean = torch.mean(torch.stack(pred_loss_list))

        return LossDict({"prediction": pred_loss_mean, "reconstruction": reconst_loss})


@dataclass
class ChimeraDataset(Dataset):
    image_type: Type[ImageBase]
    image_seqs: List[List[ImageBase]]
    vector_seqs_list: List[List[np.ndarray]]

    def __post_init__(self):
        for image_seq, vector_seqs in zip(self.image_seqs, self.vector_seqs_list):
            for vector_seq in vector_seqs:
                assert len(image_seq) == len(vector_seq)

    def __len__(self) -> int:
        return len(self.image_seqs)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        image_seq = self.image_seqs[idx]
        image_seq_tensor = torch.stack([img.to_tensor() for img in image_seq], dim=0)

        vector_seqs = self.vector_seqs_list[
            idx
        ]  # list of randomized vector originated from the same vector
        vector_seqs_tensor = torch.from_numpy(np.array(vector_seqs)).float()
        return image_seq_tensor, vector_seqs_tensor

    @classmethod
    def from_chunk(cls, chunk: MultiEpisodeChunk, encoding_rule: EncodingRule) -> "ChimeraDataset":

        first_elem_type = encoding_rule.encode_order[0]
        assert issubclass(
            first_elem_type, ImageBase
        )  # TODO(HiroIshida) relax this. see the model loss func
        image_type: Type[ImageBase] = first_elem_type
        encoding_rule.delete(image_type)  # because image encoding is done in the chimera model
        vector_seqs: List[np.ndarray] = encoding_rule.apply_to_multi_episode_chunk(chunk)

        image_seqs: List[List[ImageBase]] = []
        for episode_data in chunk:
            tmp = episode_data.get_sequence_by_type(image_type)
            image_seqs.append(tmp.elem_list)

        # data augmentation
        config = SequenceDatasetConfig()
        augmentor = SequenceDataAugmentor.from_seqs(vector_seqs, config)
        auged_vector_seqs_list = [augmentor.apply(seq) for seq in vector_seqs]

        # align seq list
        n_after_termination = 20  # TODO(HiroIshida) from config
        aligner = PaddingSequenceAligner.from_seqs(image_seqs, n_after_termination)

        image_seqs_aligned = [aligner.apply(seq) for seq in image_seqs]

        alinged_vector_seqs_list: List[List[np.ndarray]] = []
        for auged_vector_seq in auged_vector_seqs_list:
            alinged_vector_seqs_list.append([aligner.apply(seq) for seq in auged_vector_seq])
        return cls(image_type, image_seqs_aligned, alinged_vector_seqs_list)  # type: ignore
