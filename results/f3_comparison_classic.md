# FLAME 3 comparison (classic degradation, x4)

Test split: CV subset (DJI M30T), models trained on the NADIR plots (Autel XT709). n=738 frames, 738 clusters.

| metric | bicubic | edsr_lite | rrdb | swinir_lite | esrgan |
|---|---|---|---|---|---|
| PSNR (dB) | 45.068 | 47.417 | 47.509 | 47.323 | 46.437 |
| SSIM | 0.986 | 0.992 | 0.992 | 0.991 | 0.990 |
| LPIPS | 0.095 | 0.051 | 0.046 | 0.051 | 0.032 |
| M0 RMSE (K) | 4.801 | 3.792 | 3.733 | 3.837 | 4.256 |
| M0 max err (K) | 158.503 | 117.013 | 115.306 | 119.523 | 129.211 |
| M1 IoU@10K | 0.855 | 0.879 | 0.886 | 0.876 | 0.861 |
| M1 peak err (K) | 56.136 | 21.456 | 21.099 | 25.990 | 29.735 |
| M2 halluc/frame | 0.000 | 0.000 | 0.001 | 0.000 | 0.000 |
| M2 miss rate | 0.220 | 0.174 | 0.167 | 0.170 | 0.155 |
| M3 ordering rho | 0.848 | 0.891 | 0.907 | 0.890 | 0.905 |
| M4 texture ratio | 0.421 | 0.632 | 0.515 | 0.557 | 0.682 |
| M5 gradient ratio | 0.609 | 0.743 | 0.751 | 0.748 | 0.778 |
| M6 delta Hurst | 0.491 | 0.140 | 0.327 | 0.223 | 0.036 |
| M6 box dimension | 1.892 | 2.242 | 2.055 | 2.159 | 2.346 |
| M7 texture amp | 0.279 | 0.583 | 0.412 | 0.456 | 0.625 |
| M7 texture corr | 0.221 | 0.160 | 0.195 | 0.185 | 0.128 |
