import torch.nn as nn
import torch.nn.functional as F


def interpolated_base(x, scale):
    return F.interpolate(x, scale_factor=scale, mode="bicubic", align_corners=False)


def zero_init(conv):
    nn.init.zeros_(conv.weight)
    if conv.bias is not None:
        nn.init.zeros_(conv.bias)
    return conv
