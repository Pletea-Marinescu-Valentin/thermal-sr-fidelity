# Model comparison (classic degradation, x4)

Test split, cluster-bootstrap 95% CI over videos. n_frames=1144, videos=17.

| metric | bicubic | EDSR-lite (L1) | RRDB (L1) | SwinIR-lite (L1) | ESRGAN (adversarial) |
|---|---|---|---|---|---|
| PSNR (dB) | 50.264 | 53.169 | 53.170 | 53.179 | 50.964 |
| SSIM | 0.991 | 0.995 | 0.995 | 0.995 | 0.992 |
| LPIPS | 0.094 | 0.033 | 0.034 | 0.034 | 0.015 |
| M0 RMSE (K) | 0.591 | 0.474 | 0.474 | 0.473 | 0.580 |
| M0 max err (K) | 25.139 | 18.761 | 18.662 | 18.860 | 19.133 |
| M1 IoU@10K | 0.784 | 0.853 | 0.852 | 0.848 | 0.821 |
| M1 peak err (K) | 16.632 | 12.239 | 12.129 | 12.535 | 11.681 |
| M2 halluc/frame | 0.000 | 0.002 | 0.002 | 0.001 | 0.003 |
| M2 miss rate | 0.244 | 0.137 | 0.144 | 0.147 | 0.164 |
| M3 ordering rho | 0.715 | 0.875 | 0.873 | 0.872 | 0.846 |
| M3 top-1 kept | 0.765 | 0.905 | 0.904 | 0.899 | 0.873 |
| M4 texture ratio | 0.434 | 0.423 | 0.423 | 0.425 | 1.113 |
| M5 gradient ratio | 0.584 | 0.779 | 0.772 | 0.777 | 0.834 |
