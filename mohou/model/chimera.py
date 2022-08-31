import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Generic, Optional, Tuple, Type, Union

import torch

from mohou.encoder import ImageEncoder
from mohou.encoding_rule import CovarianceBasedScaleBalancer
from mohou.model import LSTM, AutoEncoderConfig, LSTMConfig
from mohou.model.autoencoder import VariationalAutoEncoder
from mohou.model.common import LossDict, ModelBase, ModelConfigBase
from mohou.trainer import TrainCache
from mohou.types import ImageT
from mohou.utils import change_color_to_yellow

logger = logging.getLogger(__name__)


@dataclass
class ChimeraConfig(ModelConfigBase):
    lstm_config: Union[Path, LSTMConfig]
    ae_config: Union[Path, AutoEncoderConfig]


class Chimera(ModelBase[ChimeraConfig], Generic[ImageT]):
    """Chimera model with lstm and autoencoder
    This is experimental model and the interface will probably be changed later.
    """

    image_type: Type[ImageT]
    lstm: LSTM
    ae: VariationalAutoEncoder[ImageT]
    balancer: Optional[CovarianceBasedScaleBalancer] = None

    def _setup_from_config(self, config: ChimeraConfig) -> None:
        if isinstance(config.lstm_config, Path):
            tcache = TrainCache[LSTM].load_from_cache_path(config.lstm_config)
            self.lstm = tcache.best_model
            project_path = config.lstm_config.parent.parent

            json_file_path = CovarianceBasedScaleBalancer.get_json_file_path(
                project_path, create_dir=True
            )
            from mohou.default import create_default_encoding_rule

            if not json_file_path.exists():
                rule = create_default_encoding_rule(project_path)
                rule.scale_balancer.dump(project_path)

                message = "NOTE: dump scale_balancer."
                logger.warn(change_color_to_yellow(message))

            balancer = CovarianceBasedScaleBalancer.load(project_path)
            self.balancer = balancer
        elif isinstance(config.lstm_config, LSTMConfig):
            self.lstm = LSTM(config.lstm_config)
            self.balancer = None

            # If accept creation from LSTMCofing, in the propagation time
            # custom encoding_rule must be prepared. And such if-else condition
            # will become bit complex. So, under construction.
            message = "NOTE: creating LSTM from LSTMConfig.\n"
            message += "creating from LSTMConfig is supposed to called only in unittest."
            logger.warn(message)
            # raise NotImplementedError("under construction...")  # TODO
        else:
            assert False

        # TODO(HiroIshida) currently fixed to vae
        if isinstance(config.ae_config, Path):
            tcache = TrainCache[VariationalAutoEncoder].load_from_cache_path(config.ae_config)
            self.ae = tcache.best_model
        elif isinstance(config.ae_config, AutoEncoderConfig):
            self.ae = VariationalAutoEncoder(config.ae_config)
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
        image_features_at_once = self.ae.encode(images_at_once)
        image_feature_seqs = image_features_at_once.reshape(n_batch, n_seqlen, -1)

        # strong assumption ... !!!!!!!!
        # TODO(HiroIshida) tmporary assume default encoding rule order (i.e. image first)
        feature_seqs = torch.concat((image_feature_seqs, vector_seqs), dim=2)

        if self.balancer is not None:
            feature_seqs = torch.stack([self.balancer.apply(seq) for seq in feature_seqs])

        # compute lstm loss
        static_context = torch.empty(n_batch, 0).to(self.device)
        indices = torch.empty(0)  # TODO: just a dummy because indices is not used in normal LSTM
        pred_loss = self.lstm.loss((indices, feature_seqs, static_context))

        # compute reconstruction loss
        ae_loss = self.ae.loss(images_at_once) * 0.1

        loss_dict = {}
        for loss in [pred_loss, ae_loss]:
            for k, v in loss.items():
                loss_dict[k] = v
        return LossDict(loss_dict)
