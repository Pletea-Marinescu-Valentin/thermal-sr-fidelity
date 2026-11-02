# Texture-loss ablation (classic, x4)

n=738 frames / 738 frame clusters. Checkpoint runs/edsr_tex_x4_classic_flame3/last.pt at step 7500.

| metric | Bicubic | EDSR (L1) | ESRGAN | EDSR + texture loss | p vs EDSR (L1) |
|---|---|---|---|---|---|
| PSNR (dB) | 45.068 | 47.417 | 46.437 | 47.178 [46.691, 47.667] | 1.8e-19 |
| LPIPS | 0.095 | 0.051 | 0.032 | 0.038 [0.036, 0.041] | 1.4e-121 |
| M0 RMSE (K) | 4.801 | 3.792 | 4.256 | 3.828 [3.683, 3.977] | 4.0e-31 |
| M4 texture ratio | 0.421 | 0.632 | 0.682 | 1.294 [1.201, 1.401] | 9.0e-117 |
| M6 delta Hurst | 0.491 | 0.140 | 0.036 | -0.253 [-0.331, -0.179] | 1.4e-121 |
| M6 box dimension | 1.892 | 2.242 | 2.346 | 2.635 [2.625, 2.645] | 1.4e-121 |
| M7 texture amplitude | 0.279 | 0.583 | 0.625 | 1.557 [1.443, 1.686] | 8.7e-118 |
| M7 texture correspondence | 0.221 | 0.160 | 0.128 | 0.070 [0.067, 0.074] | 3.5e-118 |
