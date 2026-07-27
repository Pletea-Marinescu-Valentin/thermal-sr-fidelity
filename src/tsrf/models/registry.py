MODEL_SPECS = {
    "edsr_lite": {
        "builder": "edsr_lite",
        "kwargs": dict(n_feats=64, n_blocks=16),
        "run": "runs/edsr_lite_x4_classic",
        "checkpoint": "best.pt",
        "family": "regression",
        "label": "EDSR-lite (L1)",
    },
    "swinir_lite": {
        "builder": "swinir_lite",
        "kwargs": dict(embed_dim=60, depths=(4, 4, 4, 4), num_heads=(4, 4, 4, 4)),
        "run": "runs/swinir_lite_x4_classic",
        "checkpoint": "best.pt",
        "family": "transformer",
        "label": "SwinIR-lite (L1)",
    },
    "rrdb": {
        "builder": "rrdb",
        # n_blocks=16 matches the checkpoint already on disk (body.0..15). The GAN
        # generator below must stay identical, since it is initialised from these
        # weights.
        "kwargs": dict(n_feats=64, n_blocks=16),
        "run": "runs/rrdb_x4_classic",
        "checkpoint": "best.pt",
        "family": "regression",
        "label": "RRDB (L1)",
    },
    "esrgan": {
        "builder": "rrdb",   # the GAN shares the RRDB generator -- keep kwargs equal
        "kwargs": dict(n_feats=64, n_blocks=16),
        "run": "runs/esrgan_x4_classic",
        "checkpoint": "last.pt",   # GAN trainer saves last.pt, not best.pt
        "family": "gan",
        "label": "ESRGAN (adversarial)",
    },
}

#: Order used in comparison tables: floor, then increasingly generative.
DISPLAY_ORDER = ["bicubic", "edsr_lite", "rrdb", "swinir_lite", "esrgan"]


def available_models(specs=MODEL_SPECS):
    from pathlib import Path
    return [name for name, s in specs.items()
            if (Path(s["run"]) / s["checkpoint"]).exists()]
