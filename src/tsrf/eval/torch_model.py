import numpy as np
import torch


def load_checkpoint(model, path, device="cuda"):
    ck = torch.load(path, map_location=device, weights_only=False)
    state = ck.get("model", ck)
    model.load_state_dict(state)
    model.to(device).eval()
    return model, ck.get("step"), ck.get("best_psnr")


def torch_upsampler(model, scaler, device="cuda", amp_dtype=torch.bfloat16,
                    clamp=True):
    model.eval()
    dev = torch.device(device)

    @torch.no_grad()
    def run(lr_counts):
        x = scaler.normalize(np.asarray(lr_counts, dtype=np.float32))
        t = torch.from_numpy(x)[None, None].to(dev)
        with torch.autocast("cuda", dtype=amp_dtype, enabled=(dev.type == "cuda")):
            y = model(t)
        y = y.float()
        if clamp:
            y = y.clamp(0.0, 1.0)
        return scaler.denormalize(y[0, 0].cpu().numpy().astype(np.float32))

    return run
