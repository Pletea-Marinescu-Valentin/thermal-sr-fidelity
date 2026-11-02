# FLAME 3 comparison (realistic degradation, x4)

Test split: CV subset (DJI M30T), models trained on the NADIR plots (Autel XT709). n=738 frames, 738 clusters.

| metric | bicubic | edsr_lite | rrdb | swinir_lite | esrgan |
|---|---|---|---|---|---|
| PSNR (dB) | 44.723 | 47.100 | 47.140 | 47.046 | 46.380 |
| SSIM | 0.985 | 0.991 | 0.991 | 0.991 | 0.990 |
| LPIPS | 0.101 | 0.059 | 0.055 | 0.058 | 0.039 |
| M0 RMSE (K) | 4.998 | 3.911 | 3.871 | 3.939 | 4.272 |
| M0 max err (K) | 164.072 | 121.750 | 120.517 | 124.477 | 133.852 |
| M1 IoU@10K | 0.851 | 0.875 | 0.881 | 0.874 | 0.869 |
| M1 peak err (K) | 64.832 | 32.529 | 31.235 | 35.826 | 42.262 |
| M2 halluc/frame | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| M2 miss rate | 0.249 | 0.208 | 0.203 | 0.202 | 0.180 |
| M3 ordering rho | 0.832 | 0.871 | 0.887 | 0.875 | 0.890 |
| M4 texture ratio | 0.431 | 0.627 | 0.517 | 0.555 | 0.672 |
| M5 gradient ratio | 0.577 | 0.688 | 0.690 | 0.692 | 0.713 |
| M6 delta Hurst | 0.420 | 0.137 | 0.317 | 0.220 | 0.041 |
| M6 box dimension | 1.962 | 2.245 | 2.065 | 2.162 | 2.342 |
| M7 texture amp | 0.274 | 0.574 | 0.405 | 0.447 | 0.604 |
| M7 texture corr | 0.199 | 0.144 | 0.176 | 0.166 | 0.117 |
