# Model comparison (realistic degradation, x4)

Test split, cluster-bootstrap 95% CI over videos. n_frames=1144, videos=17.

| metric | bicubic | EDSR-lite (L1) | RRDB (L1) | SwinIR-lite (L1) | ESRGAN (adversarial) |
|---|---|---|---|---|---|
| PSNR (dB) | 49.886 | 52.678 | 52.660 | 52.672 | 50.911 |
| SSIM | 0.991 | 0.995 | 0.995 | 0.995 | 0.992 |
| LPIPS | 0.102 | 0.045 | 0.045 | 0.046 | 0.021 |
| M0 RMSE (K) | 0.618 | 0.497 | 0.498 | 0.497 | 0.579 |
| M0 max err (K) | 25.923 | 20.095 | 20.034 | 20.104 | 20.034 |
| M1 IoU@10K | 0.773 | 0.838 | 0.838 | 0.834 | 0.814 |
| M1 peak err (K) | 17.833 | 14.115 | 14.194 | 14.298 | 13.227 |
| M2 halluc/frame | 0.000 | 0.002 | 0.001 | 0.001 | 0.002 |
| M2 miss rate | 0.274 | 0.178 | 0.180 | 0.185 | 0.203 |
| M3 ordering rho | 0.692 | 0.833 | 0.829 | 0.826 | 0.811 |
| M3 top-1 kept | 0.740 | 0.847 | 0.848 | 0.842 | 0.837 |
| M4 texture ratio | 0.486 | 0.492 | 0.492 | 0.493 | 1.163 |
| M5 gradient ratio | 0.555 | 0.715 | 0.710 | 0.712 | 0.765 |
