from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple, ClassVar

import torch
import torch.nn as nn

from mohou.embedder import ImageEmbedder
from mohou.model.common import LossDict, ModelBase, ModelConfigBase
from mohou.types import RGBImage, DepthImage


@dataclass
class AutoEncoderConfig(ModelConfigBase):
    n_bottleneck: int = 16
    input_shape: Tuple[int, int] = (224, 224)

    def __post_init__(self):
        # validation
        n_pixel, m_pixel = self.input_shape
        assert n_pixel == m_pixel
        assert n_pixel in [28, 112, 224]


class Reshape(nn.Module):

    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)


class AutoEncoderBase(ABC, ModelBase[AutoEncoderConfig]):
    channel: ClassVar[int]
    encoder: nn.Module
    decoder: nn.Module

    def loss(self, sample: torch.Tensor) -> LossDict:
        f_loss = nn.MSELoss()
        reconstructed = self.forward(sample)
        loss_value = f_loss(sample, reconstructed)
        return LossDict({'reconstruction': loss_value})

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(input))

    @abstractmethod
    def get_embedders(self) -> Tuple[ImageEmbedder]:
        pass

    def _create_layers(self, config: AutoEncoderConfig):
        n_pixel, m_pixel = config.input_shape

        # TODO(HiroIshida) do it programatically
        if n_pixel == 224:
            encoder = nn.Sequential(
                nn.Conv2d(self.channel, 8, 3, padding=1, stride=(2, 2)),  # 112x112
                nn.ReLU(inplace=True),
                nn.Conv2d(8, 16, 3, padding=1, stride=(2, 2)),  # 56x56
                nn.ReLU(inplace=True),
                nn.Conv2d(16, 32, 3, padding=1, stride=(2, 2)),  # 28x28
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 64, 3, padding=1, stride=(2, 2)),  # 14x14
                nn.ReLU(inplace=True),  # 64x4x4
                nn.Conv2d(64, 128, 3, padding=1, stride=(2, 2)),  # 7x7
                nn.ReLU(inplace=True),  # 64x4x4
                nn.Conv2d(128, 256, 3, padding=1, stride=(2, 2)),  # 4x4
                nn.ReLU(inplace=True),  # 64x4x4
                nn.Flatten(),
                nn.Linear(256 * 16, 512),
                nn.Linear(512, config.n_bottleneck)
            )
            decoder = nn.Sequential(
                nn.Linear(config.n_bottleneck, 512),
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
                nn.ConvTranspose2d(8, self.channel, 4, stride=2, padding=1),
                nn.Sigmoid(),
            )
        elif n_pixel == 112:
            encoder = nn.Sequential(
                nn.Conv2d(self.channel, 8, 3, padding=1, stride=(2, 2)),  # 56x56
                nn.ReLU(inplace=True),
                nn.Conv2d(8, 16, 3, padding=1, stride=(2, 2)),  # 28x28
                nn.ReLU(inplace=True),
                nn.Conv2d(16, 32, 3, padding=1, stride=(2, 2)),  # 14x14
                nn.ReLU(inplace=True),  # 64x4x4
                nn.Conv2d(32, 64, 3, padding=1, stride=(2, 2)),  # 7x7
                nn.ReLU(inplace=True),  # 64x4x4
                nn.Conv2d(64, 128, 3, padding=1, stride=(2, 2)),  # 4x4
                nn.ReLU(inplace=True),  # 64x4x4
                nn.Flatten(),
                nn.Linear(128 * 16, 512),
                nn.Linear(512, config.n_bottleneck)
            )
            decoder = nn.Sequential(
                nn.Linear(config.n_bottleneck, 512),
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
                nn.ConvTranspose2d(8, self.channel, 4, stride=2, padding=1),
                nn.Sigmoid(),
            )
        else:
            encoder = nn.Sequential(
                nn.Conv2d(self.channel, 8, 3, padding=1, stride=(2, 2)),  # 14x14
                nn.ReLU(inplace=True),
                nn.Conv2d(8, 16, 3, padding=1, stride=(2, 2)),  # 7x7
                nn.ReLU(inplace=True),  # 64x4x4
                nn.Conv2d(16, 32, 3, padding=1, stride=(2, 2)),  # 4x4
                nn.ReLU(inplace=True),  # 64x4x4
                nn.Flatten(),
                nn.Linear(32 * 16, 8 * 16),
                nn.Linear(8 * 16, config.n_bottleneck)
            )
            decoder = nn.Sequential(
                nn.Linear(config.n_bottleneck, 8 * 16),
                nn.Linear(8 * 16, 32 * 16),
                nn.ReLU(inplace=True),
                Reshape(-1, 32, 4, 4),
                nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(16, 8, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(8, self.channel, 4, stride=2, padding=1),
                nn.Sigmoid(),
            )
        self.encoder = encoder
        self.decoder = decoder


class RGBImageAutoEncoder(AutoEncoderBase):
    channel: ClassVar[int] = 3

    def get_embedders(self) -> Tuple[ImageEmbedder[RGBImage]]:
        shape = self.config.input_shape
        np_image_shape = (shape[0], shape[1], self.channel)
        embedder = ImageEmbedder[RGBImage](
            RGBImage,
            lambda image_tensor: self.encoder(image_tensor),
            lambda encoding: self.decoder(encoding),
            np_image_shape,
            self.config.n_bottleneck)
        return (embedder,)


class DepthImageAutoEncoder(AutoEncoderBase):
    channel: ClassVar[int] = 1

    def get_embedders(self) -> Tuple[ImageEmbedder[DepthImage]]:
        shape = self.config.input_shape
        np_image_shape = (shape[0], shape[1], self.channel)
        embedder = ImageEmbedder[DepthImage](
            DepthImage,
            lambda image_tensor: self.encoder(image_tensor),
            lambda encoding: self.decoder(encoding),
            np_image_shape,
            self.config.n_bottleneck)
        return (embedder,)
