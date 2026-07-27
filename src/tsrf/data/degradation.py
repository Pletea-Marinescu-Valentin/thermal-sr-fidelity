import numpy as np

#: FLIR Tau 2 noise-equivalent temperature difference, in Kelvin.
TAU2_NETD_K = 0.05


def _cubic_kernel(x, a=-0.5):
    x = np.abs(x)
    x2, x3 = x ** 2, x ** 3
    return np.where(
        x <= 1, (a + 2) * x3 - (a + 3) * x2 + 1,
        np.where(x < 2, a * x3 - 5 * a * x2 + 8 * a * x - 4 * a, 0.0),
    )


def _contributions(in_length, out_length, scale, kernel_width=4.0, antialias=True):
    if antialias and scale < 1:
        # Stretch the kernel when downscaling: this is the antialiasing step.
        kernel_width = kernel_width / scale

    x = np.arange(1, out_length + 1, dtype=np.float64)
    u = x / scale + 0.5 * (1 - 1 / scale)
    left = np.floor(u - kernel_width / 2)
    p = int(np.ceil(kernel_width) + 2)

    indices = left[:, None] + np.arange(p)[None, :]
    offsets = u[:, None] - indices
    weights = _cubic_kernel(offsets * scale) * scale if (antialias and scale < 1) \
        else _cubic_kernel(offsets)

    weights /= np.sum(weights, axis=1, keepdims=True)
    # `u` and `indices` follow MATLAB's 1-based pixel convention, so convert to
    # 0-based before indexing numpy. Skipping this shifts the result by one source
    # pixel on every pass -- about 4 HR pixels across a x4 down/up round trip, which
    # is invisible to constant/ramp/checkerboard tests because a global translation
    # preserves all three.
    indices = np.clip(indices - 1, 0, in_length - 1).astype(np.int64)

    keep = np.any(weights != 0, axis=0)
    return indices[:, keep], weights[:, keep]


def imresize_bicubic(image, scale):
    img = np.asarray(image, dtype=np.float64)
    if img.ndim != 2:
        raise ValueError(f"expected a 2-D thermal frame, got shape {img.shape}")

    in_h, in_w = img.shape
    out_h, out_w = int(np.ceil(in_h * scale)), int(np.ceil(in_w * scale))

    # Rows, then columns (separable). Index letters are named explicitly because
    # reusing one letter across differently-sized axes only happens to work on
    # square images and breaks on 640x512 frames.
    idx, wts = _contributions(in_h, out_h, scale)          # (out_h, p)
    out = np.einsum("hp,hpw->hw", wts, img[idx, :])        # -> (out_h, in_w)
    idx, wts = _contributions(in_w, out_w, scale)          # (out_w, p)
    out = np.einsum("op,hop->ho", wts, out[:, idx])        # -> (out_h, out_w)
    return out


def synthesize_lr(hr_counts, scale=4, blur_sigma=0.0, netd_k=0.0,
                  k_per_count=0.04, rng=None):
    hr = np.asarray(hr_counts, dtype=np.float64)
    if hr.ndim != 2:
        raise ValueError(f"expected a 2-D thermal frame, got shape {hr.shape}")
    if hr.shape[0] % scale or hr.shape[1] % scale:
        raise ValueError(
            f"HR size {hr.shape} is not divisible by scale {scale}; crop first so "
            "the LR/HR grids stay aligned")

    if blur_sigma > 0:
        import cv2
        ksize = int(2 * round(3 * blur_sigma) + 1)
        hr = cv2.GaussianBlur(hr, (ksize, ksize), blur_sigma,
                              borderType=cv2.BORDER_REFLECT)

    lr = imresize_bicubic(hr, 1.0 / scale)

    if netd_k > 0:
        rng = rng or np.random.default_rng()
        lr = lr + rng.normal(0.0, netd_k / k_per_count, lr.shape)
    return lr


#: Degradation presets. "classic" is the literature-standard bicubic-only setting
#: used for comparability; "realistic" adds optical blur and true sensor noise.
PRESETS = {
    "classic": dict(blur_sigma=0.0, netd_k=0.0),
    "realistic": dict(blur_sigma=0.8, netd_k=TAU2_NETD_K),
}


def degrade(hr_counts, scale=4, preset="classic", rng=None, **overrides):
    if preset not in PRESETS:
        raise ValueError(f"unknown preset {preset!r}; have {sorted(PRESETS)}")
    kwargs = {**PRESETS[preset], **overrides}
    return synthesize_lr(hr_counts, scale=scale, rng=rng, **kwargs)
