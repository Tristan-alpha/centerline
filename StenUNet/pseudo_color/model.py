"""Residual U-Net backbone for pseudo color regression."""

from __future__ import annotations

from typing import Callable, List, Tuple

import torch
from torch import Tensor, nn


def _get_conv(dimension: int) -> Callable[..., nn.Module]:
    if dimension == 2:
        return nn.Conv2d
    if dimension == 3:
        return nn.Conv3d
    raise ValueError(f"Unsupported dimension {dimension}.")


def _get_instancenorm(dimension: int) -> Callable[..., nn.Module]:
    if dimension == 2:
        return nn.InstanceNorm2d
    if dimension == 3:
        return nn.InstanceNorm3d
    raise ValueError(f"Unsupported dimension {dimension}.")


def _get_convtranspose(dimension: int) -> Callable[..., nn.Module]:
    if dimension == 2:
        return nn.ConvTranspose2d
    if dimension == 3:
        return nn.ConvTranspose3d
    raise ValueError(f"Unsupported dimension {dimension}.")


class ResidualBlock(nn.Module):
    """Simple pre-activation residual block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dimension: int,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        conv = _get_conv(dimension)
        norm = _get_instancenorm(dimension)

        self.match_channels = None
        if in_channels != out_channels:
            self.match_channels = nn.Sequential(
                conv(in_channels, out_channels, kernel_size=1),
                norm(out_channels),
            )

        self.block = nn.Sequential(
            norm(in_channels),
            nn.SiLU(inplace=True),
            conv(in_channels, out_channels, kernel_size, padding=padding),
            norm(out_channels),
            nn.SiLU(inplace=True),
            conv(out_channels, out_channels, kernel_size, padding=padding),
        )

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        if self.match_channels is not None:
            identity = self.match_channels(identity)
        return identity + self.block(x)


class DownsampleBlock(nn.Module):
    """Down-sampling block using strided convolution."""

    def __init__(self, channels: int, dimension: int) -> None:
        super().__init__()
        conv = _get_conv(dimension)
        self.layer = conv(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        return self.layer(x)


class UpsampleBlock(nn.Module):
    """Up-sampling block implemented via transposed convolution."""

    def __init__(self, in_channels: int, out_channels: int, dimension: int) -> None:
        super().__init__()
        conv_transpose = _get_convtranspose(dimension)
        self.layer = conv_transpose(
            in_channels, out_channels, kernel_size=2, stride=2
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.layer(x)


class ResUNet(nn.Module):
    """Residual U-Net backbone tailored for scalar field regression."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int = 1,
        base_channels: int = 32,
        num_stages: int = 4,
        dimension: int = 2,
    ) -> None:
        super().__init__()
        if num_stages < 2:
            raise ValueError("num_stages must be >= 2.")
        conv = _get_conv(dimension)
        norm = _get_instancenorm(dimension)

        self.stem = nn.Sequential(
            conv(in_channels, base_channels, kernel_size=3, padding=1),
            norm(base_channels),
            nn.SiLU(inplace=True),
        )

        self.encoder_blocks = nn.ModuleList()
        self.downsample_blocks = nn.ModuleList()

        channels: List[int] = []
        in_ch = base_channels
        for stage in range(num_stages):
            out_ch = base_channels * (2**stage)
            self.encoder_blocks.append(ResidualBlock(in_ch, out_ch, dimension))
            channels.append(out_ch)
            in_ch = out_ch
            if stage < num_stages - 1:
                self.downsample_blocks.append(DownsampleBlock(out_ch, dimension))

        self.decoder_blocks = nn.ModuleList()
        self.upsample_blocks = nn.ModuleList()

        for stage in reversed(range(num_stages - 1)):
            in_ch = channels[stage + 1]
            skip_ch = channels[stage]
            self.upsample_blocks.append(UpsampleBlock(in_ch, skip_ch, dimension))
            self.decoder_blocks.append(
                ResidualBlock(skip_ch * 2, skip_ch, dimension)
            )

        self.head = nn.Sequential(
            norm(channels[0]),
            nn.SiLU(inplace=True),
            conv(channels[0], out_channels, kernel_size=1),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.stem(x)
        skip_connections: List[Tensor] = []
        for encoder, down in zip(
            self.encoder_blocks[:-1], self.downsample_blocks
        ):
            x = encoder(x)
            skip_connections.append(x)
            x = down(x)
        x = self.encoder_blocks[-1](x)

        for upsample, decoder, skip in zip(
            self.upsample_blocks, self.decoder_blocks, reversed(skip_connections)
        ):
            x = upsample(x)
            if x.shape[2:] != skip.shape[2:]:
                x = torch.nn.functional.interpolate(
                    x,
                    size=skip.shape[2:],
                    mode="trilinear" if x.ndim == 5 else "bilinear",
                    align_corners=False,
                )
            x = torch.cat([x, skip], dim=1)
            x = decoder(x)
        return self.head(x)
