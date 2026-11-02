import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

from .common import interpolated_base, zero_init


class DenseBlock(nn.Module):

    def __init__(self, n_feats=64, growth=32, res_scale=0.2):
        super().__init__()
        self.res_scale = res_scale
        self.convs = nn.ModuleList([
            nn.Conv2d(n_feats + i * growth, growth if i < 4 else n_feats,
                      3, padding=1) for i in range(5)])
        self.act = nn.LeakyReLU(0.2, inplace=True)
        # Small init keeps the residual path near-identity at the start, which
        # matters for GAN stability.
        for c in self.convs:
            nn.init.kaiming_normal_(c.weight, a=0.2, mode="fan_in")
            c.weight.data *= 0.1
            nn.init.zeros_(c.bias)

    def forward(self, x):
        feats = [x]
        for i, conv in enumerate(self.convs):
            out = conv(torch.cat(feats, dim=1))
            if i < 4:
                out = self.act(out)
            feats.append(out)
        return x + feats[-1] * self.res_scale


class RRDB(nn.Module):

    def __init__(self, n_feats=64, growth=32, res_scale=0.2):
        super().__init__()
        self.blocks = nn.Sequential(*[DenseBlock(n_feats, growth, res_scale)
                                      for _ in range(3)])
        self.res_scale = res_scale

    def forward(self, x):
        return x + self.blocks(x) * self.res_scale


class RRDBNet(nn.Module):
    def __init__(self, scale=4, n_feats=64, n_blocks=12, growth=32, in_channels=1,
                 global_residual=True):
        super().__init__()
        self.scale = scale
        self.global_residual = global_residual
        self.conv_first = nn.Conv2d(in_channels, n_feats, 3, padding=1)
        self.body = nn.Sequential(*[RRDB(n_feats, growth) for _ in range(n_blocks)])
        self.conv_body = nn.Conv2d(n_feats, n_feats, 3, padding=1)

        self.up1 = nn.Conv2d(n_feats, n_feats, 3, padding=1)
        self.up2 = nn.Conv2d(n_feats, n_feats, 3, padding=1)
        self.conv_hr = nn.Conv2d(n_feats, n_feats, 3, padding=1)
        self.conv_last = nn.Conv2d(n_feats, in_channels, 3, padding=1)
        self.act = nn.LeakyReLU(0.2, inplace=True)
        if global_residual:
            zero_init(self.conv_last)

    def forward(self, x):
        base = interpolated_base(x, self.scale) if self.global_residual else 0.0
        feat = self.conv_first(x)
        feat = feat + self.conv_body(self.body(feat))
        for up in (self.up1, self.up2):
            feat = self.act(up(F.interpolate(feat, scale_factor=2, mode="nearest")))
        return base + self.conv_last(self.act(self.conv_hr(feat)))

    @property
    def n_params(self):
        return sum(p.numel() for p in self.parameters())


class UNetDiscriminator(nn.Module):

    def __init__(self, in_channels=1, n_feats=64):
        super().__init__()
        sn = spectral_norm
        self.conv0 = nn.Conv2d(in_channels, n_feats, 3, padding=1)
        self.down1 = sn(nn.Conv2d(n_feats, n_feats * 2, 4, 2, 1, bias=False))
        self.down2 = sn(nn.Conv2d(n_feats * 2, n_feats * 4, 4, 2, 1, bias=False))
        self.down3 = sn(nn.Conv2d(n_feats * 4, n_feats * 8, 4, 2, 1, bias=False))
        self.up3 = sn(nn.Conv2d(n_feats * 8, n_feats * 4, 3, 1, 1, bias=False))
        self.up2 = sn(nn.Conv2d(n_feats * 4, n_feats * 2, 3, 1, 1, bias=False))
        self.up1 = sn(nn.Conv2d(n_feats * 2, n_feats, 3, 1, 1, bias=False))
        self.conv1 = sn(nn.Conv2d(n_feats, n_feats, 3, 1, 1, bias=False))
        self.conv_last = nn.Conv2d(n_feats, 1, 3, 1, 1)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x0 = self.act(self.conv0(x))
        x1 = self.act(self.down1(x0))
        x2 = self.act(self.down2(x1))
        x3 = self.act(self.down3(x2))

        y = F.interpolate(x3, size=x2.shape[-2:], mode="bilinear", align_corners=False)
        y = self.act(self.up3(y)) + x2
        y = F.interpolate(y, size=x1.shape[-2:], mode="bilinear", align_corners=False)
        y = self.act(self.up2(y)) + x1
        y = F.interpolate(y, size=x0.shape[-2:], mode="bilinear", align_corners=False)
        y = self.act(self.up1(y)) + x0
        return self.conv_last(self.act(self.conv1(y)))

    @property
    def n_params(self):
        return sum(p.numel() for p in self.parameters())
