import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tsrf.data.dataset import (  # noqa: E402
    ThermalFrameDataset, ThermalPatchDataset, load_scaler,
)
from tsrf.models import build_model  # noqa: E402
from tsrf.models.registry import MODEL_SPECS  # noqa: E402
from tsrf.train import Trainer  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", default="edsr_lite", nargs="?")
    ap.add_argument("--steps", type=int, default=100_000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--patch", type=int, default=48, help="LR patch size")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--scale", type=int, default=4)
    ap.add_argument("--preset", default="classic")
    ap.add_argument("--netd", type=float, default=0.0,
                    help="sensor noise in K added to LR patches")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--val-every", type=int, default=2000)
    ap.add_argument("--out", default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--texture-weight", type=float, default=0.0,
                    help="weight of the M6-shaped multi-scale texture loss "
                         "(0 = plain L1, as used for every model in Table I)")
    ap.add_argument("--init-from", default=None,
                    help="initialise weights from this checkpoint, without its "
                         "optimizer or step count")
    args = ap.parse_args()

    out = Path(args.out or f"runs/{args.model}_x{args.scale}_{args.preset}")
    scaler = load_scaler()

    train_ds = ThermalPatchDataset("train", scaler=scaler, patch_lr=args.patch,
                                   scale=args.scale, preset=args.preset,
                                   netd_k=args.netd, augment=True, seed=args.seed)
    val_ds = ThermalFrameDataset("val", scaler=scaler, scale=args.scale,
                                 preset=args.preset)

    # Architecture comes from the registry, the single source of truth, so a model
    # is trained with exactly the arch it will later be evaluated (and, for RRDB,
    # fine-tuned by the GAN) with. Passing generic n_feats/n_blocks here is what
    # crashed SwinIR (which takes embed_dim/depths) and silently mismatched RRDB.
    spec = MODEL_SPECS[args.model]
    model = build_model(spec["builder"], scale=args.scale, **spec["kwargs"])
    print(f"{args.model}: {model.n_params / 1e6:.2f}M params  arch={spec['kwargs']}")
    print(f"train {len(train_ds)} frames | val {len(val_ds)} frames")
    print(f"patch {args.patch}->{args.patch * args.scale}  batch {args.batch}  "
          f"steps {args.steps}")

    milestones = tuple(int(args.steps * f) for f in (0.5, 0.8, 0.95))
    trainer = Trainer(model, train_ds, val_ds, out, lr=args.lr,
                      batch_size=args.batch, total_steps=args.steps,
                      val_every=args.val_every, ckpt_every=args.val_every,
                      milestones=milestones, num_workers=args.workers,
                      seed=args.seed, texture_weight=args.texture_weight)
    if args.texture_weight:
        print(f"texture loss enabled, weight {args.texture_weight}")
    if args.init_from:
        ck = torch.load(args.init_from, map_location="cuda", weights_only=False)
        model.load_state_dict(ck.get("model", ck))
        print(f"initialised from {args.init_from} (step {ck.get('step')})")
    if args.resume:
        trainer.resume()

    torch.backends.cudnn.benchmark = True
    trainer.train()


if __name__ == "__main__":
    main()
