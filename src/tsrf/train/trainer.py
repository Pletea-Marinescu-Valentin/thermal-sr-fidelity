import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def psnr_from_mse(mse, data_range=1.0):
    return float("inf") if mse <= 0 else 10.0 * np.log10(data_range ** 2 / mse)


class Trainer:
    def __init__(self, model, train_ds, val_ds, out_dir, lr=2e-4, batch_size=16,
                 total_steps=100_000, val_every=2000, ckpt_every=2000,
                 milestones=(50_000, 80_000, 95_000), gamma=0.5, num_workers=0,
                 device="cuda", seed=1337, amp_dtype=torch.bfloat16,
                 grad_clip=None, texture_weight=0.0, texture_scales=None):
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.out = Path(out_dir)
        self.out.mkdir(parents=True, exist_ok=True)

        set_seed(seed)
        self.train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
            pin_memory=True, drop_last=True,
            persistent_workers=bool(num_workers))
        self.val_ds = val_ds

        self.opt = torch.optim.Adam(model.parameters(), lr=lr)
        self.sched = torch.optim.lr_scheduler.MultiStepLR(
            self.opt, milestones=list(milestones), gamma=gamma)
        self.crit = nn.L1Loss()
        # Optional M6-shaped regulariser. Off by default: every model reported in
        # the paper's main table is trained with L1 alone, and this is used only
        # for the ablation of Section V-D.
        self.texture_weight = texture_weight
        self.texture = None
        if texture_weight > 0:
            from .losses import MultiScaleTextureLoss
            kw = {"scales": texture_scales} if texture_scales else {}
            self.texture = MultiScaleTextureLoss(**kw).to(self.device)

        self.total_steps = total_steps
        self.val_every = val_every
        self.ckpt_every = ckpt_every
        self.amp_dtype = amp_dtype
        self.grad_clip = grad_clip

        self.step = 0
        self.best_psnr = -float("inf")
        self.history = []

    # ------------------------------------------------------------- checkpoints

    def _state(self):
        return {"step": self.step, "model": self.model.state_dict(),
                "opt": self.opt.state_dict(), "sched": self.sched.state_dict(),
                "best_psnr": self.best_psnr, "history": self.history,
                # Recorded so a checkpoint says which objective produced it; a
                # sweep whose arms are told apart only by directory name is one
                # rename away from being mislabelled in a table.
                "texture_weight": self.texture_weight,
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all()}

    def save(self, name="last.pt"):
        tmp = self.out / (name + ".tmp")
        torch.save(self._state(), tmp)
        tmp.replace(self.out / name)   # atomic: a crash mid-write cannot corrupt

    def resume(self, name="last.pt"):
        path = self.out / name
        if not path.exists():
            return False
        ck = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ck["model"])
        self.opt.load_state_dict(ck["opt"])
        self.sched.load_state_dict(ck["sched"])
        self.step = ck["step"]
        self.best_psnr = ck["best_psnr"]
        self.history = ck.get("history", [])
        torch.set_rng_state(ck["torch_rng"].cpu().to(torch.uint8))
        if ck.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all([s.cpu().to(torch.uint8)
                                          for s in ck["cuda_rng"]])
        print(f"resumed from {path} at step {self.step}")
        return True

    # ------------------------------------------------------------- validation

    @torch.no_grad()
    def validate(self, max_frames=120):
        self.model.eval()
        total_mse, total_l1, n = 0.0, 0.0, 0
        for i in range(min(len(self.val_ds), max_frames)):
            lr, hr = self.val_ds[i]
            lr = lr[None].to(self.device)
            hr = hr[None].to(self.device)
            with torch.autocast("cuda", dtype=self.amp_dtype):
                sr = self.model(lr)
            sr = sr.float().clamp(0, 1)
            total_mse += torch.mean((sr - hr.float()) ** 2).item()
            total_l1 += torch.mean(torch.abs(sr - hr.float())).item()
            n += 1
        self.model.train()
        return {"val_psnr": psnr_from_mse(total_mse / n), "val_l1": total_l1 / n}

    # --------------------------------------------------------------- training

    def train(self):
        self.model.train()
        t0 = time.time()
        running, running_tex = [], []
        loader = iter(self.train_loader)

        while self.step < self.total_steps:
            try:
                lr_b, hr_b = next(loader)
            except StopIteration:
                loader = iter(self.train_loader)
                lr_b, hr_b = next(loader)

            lr_b = lr_b.to(self.device, non_blocking=True)
            hr_b = hr_b.to(self.device, non_blocking=True)

            with torch.autocast("cuda", dtype=self.amp_dtype):
                sr = self.model(lr_b)
                loss = self.crit(sr, hr_b)
            if self.texture is not None:
                # In float32: the loss takes a log of a square root, and bf16 has
                # too little mantissa for the small local variances of flat
                # radiometric regions.
                tex = self.texture(sr.float(), hr_b.float())
                running_tex.append(tex.item())
                loss = loss + self.texture_weight * tex

            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            if self.grad_clip:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.opt.step()
            self.sched.step()
            self.step += 1
            running.append(loss.item())

            if self.step % 200 == 0:
                rate = self.step / (time.time() - t0)
                # `running` holds the total objective, so the label must say so
                # once a second term is switched on.
                tag = "loss" if self.texture is not None else "L1"
                extra = (f"  L_tex {np.mean(running_tex[-200:]):.4f}"
                         if running_tex else "")
                print(f"  step {self.step:6d}/{self.total_steps}  "
                      f"{tag} {np.mean(running[-200:]):.5f}{extra}  "
                      f"lr {self.sched.get_last_lr()[0]:.2e}  "
                      f"{rate:.1f} it/s", flush=True)

            if self.step % self.val_every == 0:
                m = self.validate()
                m["step"] = self.step
                m["train_loss"] = float(np.mean(running[-self.val_every:]))
                if running_tex:
                    m["train_texture"] = float(
                        np.mean(running_tex[-self.val_every:]))
                self.history.append(m)
                star = ""
                if m["val_psnr"] > self.best_psnr:
                    self.best_psnr = m["val_psnr"]
                    self.save("best.pt")
                    star = "  <- best"
                print(f"  [val] step {self.step}  PSNR {m['val_psnr']:.3f} dB  "
                      f"L1 {m['val_l1']:.5f}{star}", flush=True)
                (self.out / "history.json").write_text(json.dumps(self.history, indent=1))

            if self.step % self.ckpt_every == 0:
                self.save("last.pt")

        self.save("last.pt")
        print(f"done in {(time.time() - t0) / 60:.1f} min, best val PSNR "
              f"{self.best_psnr:.3f} dB", flush=True)
        return self.history
