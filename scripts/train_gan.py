import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tsrf.data.dataset import (  # noqa: E402
    ThermalFrameDataset, ThermalPatchDataset, load_scaler,
)
from tsrf.models import RRDBNet, UNetDiscriminator  # noqa: E402
from tsrf.models.registry import MODEL_SPECS  # noqa: E402
from tsrf.train import GANTrainer  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pretrained", required=True,
                    help="L1-trained RRDB checkpoint to initialise the generator")
    ap.add_argument("--steps", type=int, default=50_000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--patch", type=int, default=32, help="LR patch size")
    ap.add_argument("--lr-g", type=float, default=1e-4)
    ap.add_argument("--lr-d", type=float, default=1e-4)
    ap.add_argument("--scale", type=int, default=4)
    ap.add_argument("--w-pixel", type=float, default=1.0)
    ap.add_argument("--w-percep", type=float, default=1.0)
    ap.add_argument("--w-gan", type=float, default=0.1)
    ap.add_argument("--no-perceptual", action="store_true")
    ap.add_argument("--preset", default="classic")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--val-every", type=int, default=2500)
    ap.add_argument("--out", default=None)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    out = Path(args.out or f"runs/esrgan_x{args.scale}_{args.preset}")
    scaler = load_scaler()

    train_ds = ThermalPatchDataset("train", scaler=scaler, patch_lr=args.patch,
                                   scale=args.scale, preset=args.preset,
                                   augment=True, seed=args.seed)
    val_ds = ThermalFrameDataset("val", scaler=scaler, scale=args.scale,
                                 preset=args.preset)

    # The generator arch must equal the RRDB it is initialised from, so both read
    # from the same registry entry ("esrgan" mirrors "rrdb").
    gen_kwargs = MODEL_SPECS["esrgan"]["kwargs"]
    gen = RRDBNet(scale=args.scale, **gen_kwargs)
    disc = UNetDiscriminator(n_feats=gen_kwargs["n_feats"])
    print(f"generator {gen.n_params / 1e6:.2f}M ({gen_kwargs}) | "
          f"discriminator {disc.n_params / 1e6:.2f}M")

    trainer = GANTrainer(
        gen, disc, train_ds, val_ds, out, lr_g=args.lr_g, lr_d=args.lr_d,
        batch_size=args.batch, total_steps=args.steps, val_every=args.val_every,
        w_pixel=args.w_pixel, w_percep=args.w_percep, w_gan=args.w_gan,
        milestones=(int(args.steps * 0.5), int(args.steps * 0.8)),
        num_workers=args.workers, seed=args.seed,
        use_perceptual=not args.no_perceptual)

    if args.resume and trainer.resume():
        pass
    else:
        trainer.load_pretrained_generator(args.pretrained)

    torch.backends.cudnn.benchmark = True
    trainer.train()


if __name__ == "__main__":
    main()
