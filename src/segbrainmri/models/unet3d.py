"""UNet 3D de référence, en PyTorch pur (aucune dépendance externe).

Architecture volontairement compacte : elle sert de base d'entraînement et de test
du pipeline. Pour la production, privilégier un modèle pré-entraîné reconnu pour la
SEP (ex. nnU-Net entraîné sur MSSEG/ISBI, LST-AI) via un adaptateur
:class:`~segbrainmri.models.base.SegmentationModel`.
"""

from __future__ import annotations

import torch
from torch import nn

from segbrainmri.models.base import SegmentationModel
from segbrainmri.models.registry import register_model


class _ConvBlock(nn.Sequential):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__(
            nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
        )


@register_model("unet3d")
class UNet3D(SegmentationModel):
    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 1,
        base_features: int = 16,
        depth: int = 4,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        feats = [base_features * 2**i for i in range(depth)]

        self.encoders = nn.ModuleList()
        ch = in_channels
        for f in feats:
            self.encoders.append(_ConvBlock(ch, f))
            ch = f
        self.pool = nn.MaxPool3d(2)

        self.ups = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for f in reversed(feats[:-1]):
            self.ups.append(nn.ConvTranspose3d(ch, f, 2, stride=2))
            self.decoders.append(_ConvBlock(2 * f, f))
            ch = f
        self.head = nn.Conv3d(ch, num_classes, 1)

    @property
    def divisor(self) -> int:
        """Les dimensions spatiales d'entrée doivent être multiples de cette valeur."""
        return 2 ** (len(self.encoders) - 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        for i, enc in enumerate(self.encoders):
            x = enc(x)
            if i < len(self.encoders) - 1:
                skips.append(x)
                x = self.pool(x)
        for up, dec, skip in zip(self.ups, self.decoders, reversed(skips), strict=True):
            x = dec(torch.cat([up(x), skip], dim=1))
        return self.head(x)
