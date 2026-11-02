import math

import torch
import torch.nn as nn

from .common import interpolated_base, zero_init


def window_partition(x, ws):
    b, h, w, c = x.shape
    x = x.view(b, h // ws, ws, w // ws, ws, c)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, ws, ws, c)


def window_reverse(windows, ws, h, w):
    b = int(windows.shape[0] / (h * w / ws / ws))
    x = windows.view(b, h // ws, w // ws, ws, ws, -1)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, h, w, -1)


class WindowAttention(nn.Module):

    def __init__(self, dim, window_size, num_heads):
        super().__init__()
        self.dim, self.ws, self.num_heads = dim, window_size, num_heads
        self.scale = (dim // num_heads) ** -0.5

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) ** 2, num_heads))
        coords = torch.stack(torch.meshgrid(
            torch.arange(window_size), torch.arange(window_size), indexing="ij"))
        coords = torch.flatten(coords, 1)
        rel = coords[:, :, None] - coords[:, None, :]
        rel = rel.permute(1, 2, 0).contiguous()
        rel[:, :, 0] += window_size - 1
        rel[:, :, 1] += window_size - 1
        rel[:, :, 0] *= 2 * window_size - 1
        self.register_buffer("relative_position_index", rel.sum(-1))

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x, mask=None):
        b_, n, c = x.shape
        qkv = self.qkv(x).reshape(b_, n, 3, self.num_heads, c // self.num_heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)

        attn = (q * self.scale) @ k.transpose(-2, -1)
        bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)].view(n, n, -1)
        attn = attn + bias.permute(2, 0, 1).contiguous().unsqueeze(0)

        if mask is not None:
            nw = mask.shape[0]
            attn = attn.view(b_ // nw, nw, self.num_heads, n, n) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, n, n)

        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(b_, n, c)
        return self.proj(x)


class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, window_size=8, shift_size=0, mlp_ratio=2.0):
        super().__init__()
        self.dim, self.ws, self.shift = dim, window_size, shift_size
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(),
                                 nn.Linear(hidden, dim))

    def _attn_mask(self, h, w, device):
        img_mask = torch.zeros((1, h, w, 1), device=device)
        cnt = 0
        for hs in (slice(0, -self.ws), slice(-self.ws, -self.shift),
                   slice(-self.shift, None)):
            for wsl in (slice(0, -self.ws), slice(-self.ws, -self.shift),
                        slice(-self.shift, None)):
                img_mask[:, hs, wsl, :] = cnt
                cnt += 1
        mw = window_partition(img_mask, self.ws).view(-1, self.ws * self.ws)
        mask = mw.unsqueeze(1) - mw.unsqueeze(2)
        return mask.masked_fill(mask != 0, -100.0).masked_fill(mask == 0, 0.0)

    def forward(self, x, size):
        h, w = size
        b, L, c = x.shape
        shortcut = x
        x = self.norm1(x).view(b, h, w, c)

        if self.shift > 0:
            x = torch.roll(x, (-self.shift, -self.shift), dims=(1, 2))
            mask = self._attn_mask(h, w, x.device)
        else:
            mask = None

        xw = window_partition(x, self.ws).view(-1, self.ws * self.ws, c)
        xw = self.attn(xw, mask)
        x = window_reverse(xw.view(-1, self.ws, self.ws, c), self.ws, h, w)

        if self.shift > 0:
            x = torch.roll(x, (self.shift, self.shift), dims=(1, 2))

        x = shortcut + x.view(b, h * w, c)
        return x + self.mlp(self.norm2(x))


class RSTB(nn.Module):

    def __init__(self, dim, depth, num_heads, window_size=8, mlp_ratio=2.0):
        super().__init__()
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim, num_heads, window_size,
                                 0 if i % 2 == 0 else window_size // 2, mlp_ratio)
            for i in range(depth)])
        self.conv = nn.Conv2d(dim, dim, 3, padding=1)

    def forward(self, x, size):
        h, w = size
        res = x
        for blk in self.blocks:
            x = blk(x, size)
        b, L, c = x.shape
        x = x.transpose(1, 2).view(b, c, h, w)
        x = self.conv(x).flatten(2).transpose(1, 2)
        return x + res


class SwinIRLite(nn.Module):
    def __init__(self, scale=4, embed_dim=60, depths=(4, 4, 4, 4),
                 num_heads=(4, 4, 4, 4), window_size=8, mlp_ratio=2.0,
                 in_channels=1, global_residual=True):
        super().__init__()
        self.scale = scale
        self.window_size = window_size
        self.global_residual = global_residual

        self.conv_first = nn.Conv2d(in_channels, embed_dim, 3, padding=1)
        self.layers = nn.ModuleList([
            RSTB(embed_dim, d, nh, window_size, mlp_ratio)
            for d, nh in zip(depths, num_heads)])
        self.norm = nn.LayerNorm(embed_dim)
        self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, padding=1)

        ups = []
        for _ in range(int(math.log2(scale))):
            ups += [nn.Conv2d(embed_dim, 4 * embed_dim, 3, padding=1),
                    nn.PixelShuffle(2)]
        self.upsample = nn.Sequential(*ups)
        self.conv_last = nn.Conv2d(embed_dim, in_channels, 3, padding=1)
        if global_residual:
            zero_init(self.conv_last)

    def _pad(self, x):
        _, _, h, w = x.shape
        ph = (self.window_size - h % self.window_size) % self.window_size
        pw = (self.window_size - w % self.window_size) % self.window_size
        if ph or pw:
            x = nn.functional.pad(x, (0, pw, 0, ph), mode="reflect")
        return x, h, w

    def forward(self, x):
        base = interpolated_base(x, self.scale) if self.global_residual else None
        x, h0, w0 = self._pad(x)
        feat = self.conv_first(x)

        b, c, h, w = feat.shape
        t = feat.flatten(2).transpose(1, 2)
        for layer in self.layers:
            t = layer(t, (h, w))
        t = self.norm(t).transpose(1, 2).view(b, c, h, w)

        feat = feat + self.conv_after_body(t)
        out = self.conv_last(self.upsample(feat))
        out = out[:, :, :h0 * self.scale, :w0 * self.scale]
        return out if base is None else base + out

    @property
    def n_params(self):
        return sum(p.numel() for p in self.parameters())
