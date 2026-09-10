"""Positive control: does the protocol separate fabrication from attenuation?

Every claim in this work rests on M4, M6 and M7 meaning what we say they mean.
On model outputs that cannot be checked directly, because the "right" answer is
unknown. So we build predictions from real radiometric frames whose answer *is*
known by construction, and ask the protocol to tell them apart. All three keep
the reference's low-frequency content exactly and differ only in fine texture:

  attenuated         keeps the measured texture, scaled down   (the L1 failure)
  fabricated_global  replaces it with a phase-randomised surrogate
  fabricated_local   the same surrogate, rescaled to the reference's local
                     amplitude                                 (the GAN failure)

Phase randomisation is the standard surrogate: multiplying the reference's
amplitude spectrum by the phases of an independent Gaussian field gives a real
field with the *same* power spectrum and no spatial relation to the original --
an exact instance of "statistically correct, in the wrong place".

The two fabricated arms differ in a way that turns out to matter. Real thermal
texture is spatially inhomogeneous, busy near objects and quiet on a road, so a
surrogate that matches the spectrum of the whole frame spreads that energy
uniformly and overshoots by roughly six times in exactly the flat regions the
protocol examines. Matching the local amplitude instead is both the fairer
control and the better model of a generator producing locally plausible
texture. The gap between the two arms is also the mechanism behind the
overshoot of the texture objective in the ablation.

A protocol that cannot separate `attenuated` from `fabricated_local` has not
measured hallucination. The expected outcome is that M4 and M6 rank the
fabricated field as the more faithful of the two while M7 rejects it, and that
the trained models fall on the controls: the regressors near `attenuated`, the
adversarial model near `fabricated_local`.
"""

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import uniform_filter

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tsrf.data.radiometry import raw_to_kelvin  # noqa: E402
from tsrf.eval.runner import load_splits  # noqa: E402
from tsrf.metrics import (  # noqa: E402
    cold_region_smoothness, texture_correspondence, texture_scaling,
)
from tsrf.metrics.fractal import FLAT_WINDOW  # noqa: E402
from tsrf.stats import cluster_bootstrap_ci  # noqa: E402

SPLIT_DIR = Path("data/adas/images_thermal_val/analyticsData")
OUT = Path("results/positive_control.json")
N_FRAMES = 200
ATTENUATION = 0.3


def phase_randomised(x, rng):
    """A field with the same power spectrum as `x` and no relation to it.

    Built by keeping the amplitude spectrum of `x` and taking the phases of an
    independent real Gaussian field. Because both the amplitude of a real
    field's spectrum and the phase pattern of another real field are Hermitian,
    the product is Hermitian and the inverse transform is exactly real; no
    symmetrisation by hand is needed.
    """
    amp = np.abs(np.fft.fft2(x))
    noise = np.fft.fft2(rng.standard_normal(x.shape))
    phase = noise / np.maximum(np.abs(noise), 1e-30)
    return np.real(np.fft.ifft2(amp * phase))


def local_std(x, window):
    m = uniform_filter(x, size=window, mode="reflect")
    sq = uniform_filter(x * x, size=window, mode="reflect")
    return np.sqrt(np.maximum(sq - m * m, 0.0))


def build_controls(t_gt, rng, window=FLAT_WINDOW, attenuation=ATTENUATION):
    """Three predictions with known answers, differing only in fine texture.

    `fabricated_global` matches the reference's power spectrum over the whole
    frame, which is what a global statistical objective asks for. Real thermal
    texture is spatially inhomogeneous -- busy near objects, quiet on a road --
    so spreading that energy uniformly overshoots badly in exactly the flat
    regions the protocol examines. `fabricated_local` therefore rescales the
    surrogate to the reference's *local* amplitude, which is the honest model of
    a generator that produces locally plausible texture, and the control the
    claim actually needs.
    """
    low = uniform_filter(t_gt, size=window, mode="reflect")
    fine = t_gt - low
    surrogate = phase_randomised(fine, rng)
    gain = local_std(fine, window) / np.maximum(local_std(surrogate, window), 1e-9)
    return {
        "attenuated": low + attenuation * fine,
        "fabricated_global": low + surrogate,
        "fabricated_local": low + surrogate * gain,
    }


def score(pred, gt):
    m4 = cold_region_smoothness(pred, gt)
    m6 = texture_scaling(pred, gt)
    m7 = texture_correspondence(pred, gt)
    return {
        "m4_texture_ratio": m4["texture_ratio"],
        "m6_flat_delta_hurst": m6["flat_delta_hurst"],
        "m6_flat_dim_sr": m6["flat_dim_sr"],
        "m7_flat_hf_amp_ratio": m7["flat_hf_amp_ratio"],
        "m7_flat_hf_corr": m7["flat_hf_corr"],
    }


def main():
    frames = load_splits()["test"][:N_FRAMES]
    rng = np.random.default_rng(1337)
    print(f"{len(frames)} frames, attenuation {ATTENUATION}, "
          f"surrogate = phase randomisation\n")

    recs = {"attenuated": [], "fabricated_global": [], "fabricated_local": []}
    corr_check = []
    for i, name in enumerate(frames, 1):
        raw = np.array(Image.open(SPLIT_DIR / name), dtype=np.float64)
        gt = raw_to_kelvin(raw)
        controls = build_controls(gt, rng)
        for arm, pred in controls.items():
            recs[arm].append(score(pred, gt))

        # Whole-frame correlation of the fabricated arm's high-pass with the
        # reference's. This is an upper bound on any residual relation, not a
        # measure of the surrogate itself: it also picks up the low-frequency
        # leakage of a box high-pass at object boundaries, and the fact that the
        # local gain is a function of the reference. The quantity that matters
        # for the claim is the same correlation restricted to the flat set,
        # which is M7 and is reported in the table below.
        fine = gt - uniform_filter(gt, size=FLAT_WINDOW, mode="reflect")
        sur = controls["fabricated_local"] - uniform_filter(
            controls["fabricated_local"], size=FLAT_WINDOW, mode="reflect")
        corr_check.append(abs(float(np.corrcoef(fine.ravel(), sur.ravel())[0, 1])))
        if i % 50 == 0:
            print(f"  {i}/{len(frames)}")

    clusters = np.arange(len(frames))
    agg = {arm: {k: cluster_bootstrap_ci([r[k] for r in rs], clusters, n_boot=2000)
                 for k in rs[0]} for arm, rs in recs.items()}

    print(f"\nwhole-frame upper bound on residual correlation: "
          f"{np.mean(corr_check):.4f}  (the flat-set value is M7, below)\n")

    keys = ["m4_texture_ratio", "m6_flat_delta_hurst", "m6_flat_dim_sr",
            "m7_flat_hf_amp_ratio", "m7_flat_hf_corr"]
    arms = ["attenuated", "fabricated_global", "fabricated_local"]
    print(f"{'metric':<26}" + "".join(f"{a:>20}" for a in arms))
    print("-" * 86)
    for k in keys:
        row = "".join(f"{agg[a][k]['mean']:>12.3f}"
                      f" [{agg[a][k]['lo']:>3.2f}]" for a in arms)
        print(f"{k:<26}{row}")

    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps({
        "n_frames": len(frames), "attenuation": ATTENUATION,
        "residual_correlation_upper_bound": float(np.mean(corr_check)),
        "aggregate": agg}, indent=1))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
