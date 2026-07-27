import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .trainer import psnr_from_mse, set_seed

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class VGGPerceptualLoss(nn.Module):

    def __init__(self, layer="conv5_4", weight=1.0, device="cuda"):
        super().__init__()
        from torchvision.models import VGG19_Weights, vgg19
        vgg = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features
        # conv5_4 output, before the final ReLU, is index 34 in torchvision's VGG19.
        self.slice = nn.Sequential(*list(vgg.children())[:35]).eval().to(device)
        for p in self.slice.parameters():
            p.requires_grad_(False)
        self.weight = weight
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))
        self.to(device)

    def _prep(self, x):
        x = x.clamp(0, 1).repeat(1, 3, 1, 1)
        return (x - self.mean) / self.std

    def forward(self, sr, hr):
        return self.weight * nn.functional.l1_loss(
            self.slice(self._prep(sr)), self.slice(self._prep(hr)))


class GANTrainer:
    def __init__(self, gen, disc, train_ds, val_ds, out_dir, lr_g=1e-4, lr_d=1e-4,
                 batch_size=8, total_steps=50_000, val_every=2000,
                 w_pixel=1.0, w_percep=1.0, w_gan=0.1, milestones=(25_000, 40_000),
                 gamma=0.5, num_workers=0, device="cuda", seed=1337,
                 amp_dtype=torch.bfloat16, use_perceptual=True):
        self.device = torch.device(device)
        self.gen = gen.to(self.device)
        self.disc = disc.to(self.device)
        self.out = Path(out_dir)
        self.out.mkdir(parents=True, exist_ok=True)

        set_seed(seed)
        self.train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
            pin_memory=True, drop_last=True, persistent_workers=bool(num_workers))
        self.val_ds = val_ds

        self.opt_g = torch.optim.Adam(gen.parameters(), lr=lr_g, betas=(0.9, 0.99))
        self.opt_d = torch.optim.Adam(disc.parameters(), lr=lr_d, betas=(0.9, 0.99))
        self.sched_g = torch.optim.lr_scheduler.MultiStepLR(
            self.opt_g, list(milestones), gamma)
        self.sched_d = torch.optim.lr_scheduler.MultiStepLR(
            self.opt_d, list(milestones), gamma)

        self.l1 = nn.L1Loss()
        self.bce = nn.BCEWithLogitsLoss()
        self.percep = VGGPerceptualLoss(device=device) if use_perceptual else None

        self.w_pixel, self.w_percep, self.w_gan = w_pixel, w_percep, w_gan
        self.total_steps, self.val_every = total_steps, val_every
        self.amp_dtype = amp_dtype
        self.step = 0
        self.history = []

    # ------------------------------------------------------------- checkpoints

    def save(self, name="last.pt"):
        state = {"step": self.step, "model": self.gen.state_dict(),
                 "disc": self.disc.state_dict(), "opt_g": self.opt_g.state_dict(),
                 "opt_d": self.opt_d.state_dict(),
                 "sched_g": self.sched_g.state_dict(),
                 "sched_d": self.sched_d.state_dict(), "history": self.history}
        tmp = self.out / (name + ".tmp")
        torch.save(state, tmp)
        tmp.replace(self.out / name)

    def resume(self, name="last.pt"):
        path = self.out / name
        if not path.exists():
            return False
        ck = torch.load(path, map_location=self.device, weights_only=False)
        self.gen.load_state_dict(ck["model"])
        self.disc.load_state_dict(ck["disc"])
        self.opt_g.load_state_dict(ck["opt_g"])
        self.opt_d.load_state_dict(ck["opt_d"])
        self.sched_g.load_state_dict(ck["sched_g"])
        self.sched_d.load_state_dict(ck["sched_d"])
        self.step = ck["step"]
        self.history = ck.get("history", [])
        print(f"resumed GAN from {path} at step {self.step}", flush=True)
        return True

    def load_pretrained_generator(self, path):
        ck = torch.load(path, map_location=self.device, weights_only=False)
        self.gen.load_state_dict(ck.get("model", ck))
        print(f"generator initialised from {path} (step {ck.get('step')})", flush=True)

    # ------------------------------------------------------------- validation

    @torch.no_grad()
    def validate(self, max_frames=120):
        self.gen.eval()
        mse = l1 = 0.0
        n = 0
        for i in range(min(len(self.val_ds), max_frames)):
            lr, hr = self.val_ds[i]
            lr, hr = lr[None].to(self.device), hr[None].to(self.device)
            with torch.autocast("cuda", dtype=self.amp_dtype):
                sr = self.gen(lr)
            sr = sr.float().clamp(0, 1)
            mse += torch.mean((sr - hr.float()) ** 2).item()
            l1 += torch.mean(torch.abs(sr - hr.float())).item()
            n += 1
        self.gen.train()
        return {"val_psnr": psnr_from_mse(mse / n), "val_l1": l1 / n}

    # --------------------------------------------------------------- training

    def train(self):
        self.gen.train()
        self.disc.train()
        t0 = time.time()
        loader = iter(self.train_loader)
        run_g, run_d = [], []

        while self.step < self.total_steps:
            try:
                lr_b, hr_b = next(loader)
            except StopIteration:
                loader = iter(self.train_loader)
                lr_b, hr_b = next(loader)
            lr_b = lr_b.to(self.device, non_blocking=True)
            hr_b = hr_b.to(self.device, non_blocking=True)

            # ---- generator
            for p in self.disc.parameters():
                p.requires_grad_(False)
            with torch.autocast("cuda", dtype=self.amp_dtype):
                sr = self.gen(lr_b)
                loss_g = self.w_pixel * self.l1(sr, hr_b)
                if self.percep is not None:
                    loss_g = loss_g + self.w_percep * self.percep(sr.float(), hr_b.float())
                fake_logit = self.disc(sr)
                loss_g = loss_g + self.w_gan * self.bce(
                    fake_logit, torch.ones_like(fake_logit))
            self.opt_g.zero_grad(set_to_none=True)
            loss_g.backward()
            self.opt_g.step()

            # ---- discriminator
            for p in self.disc.parameters():
                p.requires_grad_(True)
            with torch.autocast("cuda", dtype=self.amp_dtype):
                real_logit = self.disc(hr_b)
                loss_real = self.bce(real_logit, torch.ones_like(real_logit))
                fake_logit = self.disc(sr.detach())
                loss_fake = self.bce(fake_logit, torch.zeros_like(fake_logit))
                loss_d = 0.5 * (loss_real + loss_fake)
            self.opt_d.zero_grad(set_to_none=True)
            loss_d.backward()
            self.opt_d.step()

            self.sched_g.step()
            self.sched_d.step()
            self.step += 1
            run_g.append(loss_g.item())
            run_d.append(loss_d.item())

            if self.step % 200 == 0:
                print(f"  step {self.step:6d}/{self.total_steps}  "
                      f"G {np.mean(run_g[-200:]):.4f}  D {np.mean(run_d[-200:]):.4f}  "
                      f"{self.step / (time.time() - t0):.1f} it/s", flush=True)

            if self.step % self.val_every == 0:
                m = self.validate()
                m["step"] = self.step
                m["loss_g"] = float(np.mean(run_g[-self.val_every:]))
                m["loss_d"] = float(np.mean(run_d[-self.val_every:]))
                self.history.append(m)
                print(f"  [val] step {self.step}  PSNR {m['val_psnr']:.3f} dB",
                      flush=True)
                (self.out / "history.json").write_text(json.dumps(self.history, indent=1))
                self.save("last.pt")

        self.save("last.pt")
        print(f"GAN done in {(time.time() - t0) / 60:.1f} min", flush=True)
        return self.history
