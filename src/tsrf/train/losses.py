"""Differentiable texture-scaling loss.

The metric side of this work (M6) fits a power law to the local standard
deviation across window sizes. The same ladder can be made an objective: rather
than constraining a reconstruction with a heat-transfer model, which needs
boundary conditions the data does not carry, we constrain the scale-invariant
statistics of the field, which are read directly off the reference.

    L_tex = (1/|W|) * sum_w  | log mean sigma_w(SR)  -  log mean sigma_w(HR) |

Matching every rung of the ladder constrains the amplitude and the exponent at
once: a constant offset in log space is an amplitude error, a tilt is an
exponent error. The whole thing is box filters and is therefore cheap.

What it cannot do is make the emitted texture correspond to the measured one --
it is a statistic of each image separately, so it is blind to whether the two
match pixel for pixel. That limitation is the point of the experiment in the
paper, not an oversight: see M7.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

#: Window ladder, matching DEFAULT_SCALES on the metric side.
DEFAULT_SCALES = (3, 5, 9, 17)


def local_std(x, window, eps=1e-12):
    """Local standard deviation over a `window` x `window` box, differentiable."""
    pad = window // 2
    mean = F.avg_pool2d(x, window, stride=1, padding=pad, count_include_pad=False)
    sq = F.avg_pool2d(x * x, window, stride=1, padding=pad, count_include_pad=False)
    return torch.sqrt(torch.clamp(sq - mean * mean, min=0.0) + eps)


class MultiScaleTextureLoss(nn.Module):
    """Matches the local-variance ladder of a prediction to its reference."""

    def __init__(self, scales=DEFAULT_SCALES, eps=1e-6):
        super().__init__()
        self.scales = tuple(scales)
        self.eps = eps

    def forward(self, sr, hr):
        loss = sr.new_zeros(())
        for w in self.scales:
            # Mean over space, per image and channel: the ladder rung of M6.
            s = local_std(sr, w).mean(dim=(-2, -1))
            g = local_std(hr, w).mean(dim=(-2, -1))
            loss = loss + (torch.log(s + self.eps)
                           - torch.log(g + self.eps)).abs().mean()
        return loss / len(self.scales)
