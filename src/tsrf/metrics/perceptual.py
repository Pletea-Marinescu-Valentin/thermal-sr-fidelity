import numpy as np
from skimage.metrics import structural_similarity

# LPIPS needs torch + a pretrained net, so it is imported lazily and cached. Keeping
# it out of module import means the rest of the metrics package (and its 64 tests)
# stays torch-free and runs without a GPU.
_LPIPS_MODEL = None
_LPIPS_DEVICE = None


def psnr(pred_norm, gt_norm, data_range=1.0):
    p = np.asarray(pred_norm, dtype=np.float64)
    g = np.asarray(gt_norm, dtype=np.float64)
    if p.shape != g.shape:
        raise ValueError(f"shape mismatch: {p.shape} vs {g.shape}")
    mse = float(np.mean((p - g) ** 2))
    if mse == 0:
        return float("inf")
    return float(10.0 * np.log10(data_range ** 2 / mse))


def ssim(pred_norm, gt_norm, data_range=1.0):
    p = np.asarray(pred_norm, dtype=np.float64)
    g = np.asarray(gt_norm, dtype=np.float64)
    if p.shape != g.shape:
        raise ValueError(f"shape mismatch: {p.shape} vs {g.shape}")
    return float(structural_similarity(g, p, data_range=data_range))


def _get_lpips():
    global _LPIPS_MODEL, _LPIPS_DEVICE
    if _LPIPS_MODEL is None:
        import lpips
        import torch
        _LPIPS_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        _LPIPS_MODEL = lpips.LPIPS(net="alex", verbose=False).to(_LPIPS_DEVICE).eval()
    return _LPIPS_MODEL, _LPIPS_DEVICE


def lpips_distance(pred_norm, gt_norm):
    import torch
    model, device = _get_lpips()

    def prep(x):
        x = np.clip(np.asarray(x, dtype=np.float32), 0.0, 1.0)
        t = torch.from_numpy(x)[None, None].repeat(1, 3, 1, 1) * 2.0 - 1.0
        return t.to(device)

    with torch.no_grad():
        return float(model(prep(pred_norm), prep(gt_norm)).item())


def perceptual_metrics(pred_norm, gt_norm, data_range=1.0, include_lpips=False):
    out = {"psnr_db": psnr(pred_norm, gt_norm, data_range),
           "ssim": ssim(pred_norm, gt_norm, data_range)}
    if include_lpips:
        out["lpips"] = lpips_distance(pred_norm, gt_norm)
    return out
