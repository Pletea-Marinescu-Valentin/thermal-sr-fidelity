import torch
import torch.nn as nn

from .common import interpolated_base, zero_init


class ResidualBlock(nn.Module):

    def __init__(self, n_feats, res_scale=1.0):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_feats, n_feats, 3, padding=1),
        )
        self.res_scale = res_scale

    def forward(self, x):
        return x + self.body(x) * self.res_scale


class Upsampler(nn.Sequential):

    def __init__(self, scale, n_feats):
        layers = []
        if scale & (scale - 1):
            raise ValueError(f"scale must be a power of two, got {scale}")
        for _ in range(int(torch.log2(torch.tensor(float(scale))).item())):
            layers += [nn.Conv2d(n_feats, 4 * n_feats, 3, padding=1),
                       nn.PixelShuffle(2)]
        super().__init__(*layers)


class EDSRLite(nn.Module):
    def __init__(self, scale=4, n_feats=64, n_blocks=16, res_scale=1.0,
                 in_channels=1, global_residual=True):
        super().__init__()
        self.scale = scale
        self.global_residual = global_residual
        self.head = nn.Conv2d(in_channels, n_feats, 3, padding=1)
        self.body = nn.Sequential(
            *[ResidualBlock(n_feats, res_scale) for _ in range(n_blocks)],
            nn.Conv2d(n_feats, n_feats, 3, padding=1),
        )
        self.conv_last = nn.Conv2d(n_feats, in_channels, 3, padding=1)
        self.tail = nn.Sequential(Upsampler(scale, n_feats), self.conv_last)
        if global_residual:
            zero_init(self.conv_last)

    def forward(self, x):
        base = interpolated_base(x, self.scale) if self.global_residual else 0.0
        f = self.head(x)
        f = f + self.body(f)          # long skip connection
        return base + self.tail(f)

    @property
    def n_params(self):
        return sum(p.numel() for p in self.parameters())
