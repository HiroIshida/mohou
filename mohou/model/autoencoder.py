from abc import abstractmethod
from dataclasses import dataclass
from typing import Generic, Tuple, Type, List

import numpy as np
import torch
import torch.nn as nn

from mohou.embedder import ImageEmbedder
from mohou.model.common import LossDict, ModelBase, ModelConfigBase
from mohou.types import ImageT, ImageBase


@dataclass
class AutoEncoderConfig(ModelConfigBase):
    image_type: Type[ImageBase]
    n_bottleneck: int = 16
    n_pixel: int = 112

    def __post_init__(self):
        # validation
        assert self.n_pixel in [28, 112, 224]


class Reshape(nn.Module):

    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)


def create_encoder_decoder_layers(n_channel: int, n_pixel: int, n_bottleneck: int) -> Tuple[List[nn.Module], List[nn.Module]]:
    assert n_pixel in [28, 112, 224]

    if n_pixel == 224:
        encoder_layers = [
            nn.Conv2d(n_channel, 8, 3, padding=1, stride=(2, 2)),  # 112x112
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 16, 3, padding=1, stride=(2, 2)),  # 56x56
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, padding=1, stride=(2, 2)),  # 28x28
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, padding=1, stride=(2, 2)),  # 14x14
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, padding=1, stride=(2, 2)),  # 7x7
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, padding=1, stride=(2, 2)),  # 4x4
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(256 * 16, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, n_bottleneck),
            nn.ReLU(inplace=True),
        ]
        decoder_layers = [
            nn.Linear(n_bottleneck, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256 * 16),
            nn.ReLU(inplace=True),
            Reshape(-1, 256, 4, 4),
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, 8, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(8, n_channel, 4, stride=2, padding=1),
            nn.Sigmoid(),
        ]
    elif n_pixel == 112:
        encoder_layers = [
            nn.Conv2d(n_channel, 8, 3, padding=1, stride=(2, 2)),  # 56x56
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 16, 3, padding=1, stride=(2, 2)),  # 28x28
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, padding=1, stride=(2, 2)),  # 14x14
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, padding=1, stride=(2, 2)),  # 7x7
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, padding=1, stride=(2, 2)),  # 4x4
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(128 * 16, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, n_bottleneck),
            nn.ReLU(inplace=True),
        ]
        decoder_layers = [
            nn.Linear(n_bottleneck, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 128 * 16),
            nn.ReLU(inplace=True),
            Reshape(-1, 128, 4, 4),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, 8, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(8, n_channel, 4, stride=2, padding=1),
            nn.Sigmoid(),
        ]
    else:
        encoder_layers = [
            nn.Conv2d(n_channel, 8, 3, padding=1, stride=(2, 2)),  # 14x14
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 16, 3, padding=1, stride=(2, 2)),  # 7x7
            nn.ReLU(inplace=True),  # 64x4x4
            nn.Conv2d(16, 32, 3, padding=1, stride=(2, 2)),  # 4x4
            nn.ReLU(inplace=True),  # 64x4x4
            nn.Flatten(),
            nn.Linear(32 * 16, 8 * 16),
            nn.ReLU(inplace=True),
            nn.Linear(8 * 16, n_bottleneck),
            nn.ReLU(inplace=True),
        ]
        decoder_layers = [
            nn.Linear(n_bottleneck, 8 * 16),
            nn.ReLU(inplace=True),
            nn.Linear(8 * 16, 32 * 16),
            nn.ReLU(inplace=True),
            Reshape(-1, 32, 4, 4),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, 8, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(8, n_channel, 4, stride=2, padding=1),
            nn.Sigmoid(),
        ]
    return encoder_layers, decoder_layers


class AutoEncoderBase(ModelBase[AutoEncoderConfig], Generic[ImageT]):
    image_type: Type[ImageT]
    encoder_module: nn.Module
    decoder_module: nn.Module
    n_pixel: int

    @abstractmethod
    def loss(self, sample: torch.Tensor) -> LossDict:
        pass

    @abstractmethod
    def get_encoder(self) -> nn.Module:
        """Must be deterministic """
        pass

    @abstractmethod
    def get_decoder(self) -> nn.Module:
        """Must be deterministic """
        pass

    def check_network_input(self, inp: torch.Tensor):
        assert inp.ndim == 4
        assert list(inp.shape[2:]) == [self.n_pixel, self.n_pixel]
        assert self.image_type.channel() == inp.shape[1], 'channel mismatch'

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        self.check_network_input(input)
        return self.get_decoder()(self.get_encoder()(input))

    def get_embedder(self) -> ImageEmbedder[ImageT]:
        np_image_shape = (self.config.n_pixel, self.config.n_pixel, self.channel())
        embedder = ImageEmbedder[ImageT](
            self.image_type,
            lambda image_tensor: self.get_encoder()(image_tensor),
            lambda encoding: self.get_decoder()(encoding),
            np_image_shape,
            self.config.n_bottleneck)
        return embedder

    def channel(self) -> int:
        return self.image_type.channel()


class AutoEncoder(AutoEncoderBase):

    def loss(self, sample: torch.Tensor) -> LossDict:
        self.check_network_input(sample)
        f_loss = nn.MSELoss()
        reconstructed = self.forward(sample)
        loss_value = f_loss(sample, reconstructed)
        return LossDict({'reconstruction': loss_value})

    def get_encoder(self) -> nn.Module:
        return self.encoder_module

    def get_decoder(self) -> nn.Module:
        return self.decoder_module

    def _setup_from_config(self, config: AutoEncoderConfig):
        self.image_type = config.image_type  # type: ignore
        n_pixel = config.n_pixel
        self.n_pixel = n_pixel
        encoder_layers, decoder_layers = create_encoder_decoder_layers(self.channel(), config.n_pixel, config.n_bottleneck)
        self.encoder_module = nn.Sequential(*encoder_layers)
        self.decoder_module = nn.Sequential(*decoder_layers)


class VariationalAutoEncoder(AutoEncoderBase):
    dense_mean: nn.Module
    dense_var: nn.Module

    def loss(self, sample: torch.Tensor) -> LossDict:
        self.check_network_input(sample)

        encoded = self.encoder_module(sample)
        mu = self.dense_mean(encoded)
        logvar = self.dense_var(encoded)
        z = self.reparameterize(mu, logvar)
        reconstructed = self.decoder_module(z)

        weight = 1e-1 * self.config.n_bottleneck / np.prod(sample.shape)
        kld_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim=1), dim=0) * weight
        loss_value = nn.MSELoss()(sample, reconstructed)
        return LossDict({'reconstruction': loss_value, 'kld': kld_loss})

    def get_encoder(self) -> nn.Module:
        return nn.Sequential(self.encoder_module, self.dense_mean)

    def get_decoder(self) -> nn.Module:
        return self.decoder_module

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def _setup_from_config(self, config: AutoEncoderConfig):
        self.image_type = config.image_type  # type: ignore
        n_pixel = config.n_pixel
        encoder_layers, decoder_layers = create_encoder_decoder_layers(self.channel(), config.n_pixel, config.n_bottleneck)
        encoder_layers.pop()  # remove relu
        encoder_layers.pop()  # remove dense

        encoder_last_dense: nn.Linear = encoder_layers[-2]  # type: ignore
        out_dim = encoder_last_dense.out_features

        self.encoder_module = nn.Sequential(*encoder_layers)
        self.decoder_module = nn.Sequential(*decoder_layers)
        self.dense_mean = nn.Linear(out_dim, config.n_bottleneck)
        self.dense_var = nn.Linear(out_dim, config.n_bottleneck)
        self.n_pixel = n_pixel
