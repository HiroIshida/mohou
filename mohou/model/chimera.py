from dataclasses import dataclass
from typing import Generic, Tuple, Type, Union

import torch
import torch.nn as nn

from mohou.encoder import ImageEncoder
from mohou.model import LSTM, AutoEncoderConfig, LSTMConfig
from mohou.model.autoencoder import AutoEncoder
from mohou.model.common import LossDict, ModelBase, ModelConfigBase
from mohou.types import ImageT


@dataclass
class ChimeraConfig(ModelConfigBase):
    """config for Chimera model
    note: considering the original roll of the model config, which constructdefault
    model only from the config instance, ae_config should be AutoEncoderConfig.
    However, in the chimera model it is common to first pre-train
    the autoencoder model only, and then train the entire chimera model. Thus, chimera config
    may contain AutoEncoder as the ae_config.
    """

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
