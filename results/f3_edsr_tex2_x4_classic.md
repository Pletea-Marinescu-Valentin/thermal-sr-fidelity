# Texture-loss ablation (classic, x4)

n=738 frames / 738 frame clusters. Checkpoint runs/edsr_tex2_x4_classic_flame3/last.pt at step 7500.

| metric | Bicubic | EDSR (L1) | ESRGAN | EDSR + texture loss | p vs EDSR (L1) |
|---|---|---|---|---|---|
| PSNR (dB) | 45.068 | 47.417 | 46.437 | 46.949 [46.486, 47.415] | 1.6e-29 |
| LPIPS | 0.095 | 0.051 | 0.032 | 0.039 [0.037, 0.042] | 1.4e-121 |
| M0 RMSE (K) | 4.801 | 3.792 | 4.256 | 3.848 [3.702, 3.995] | 6.2e-44 |
| M4 texture ratio | 0.421 | 0.632 | 0.682 | 1.456 [1.363, 1.561] | 9.0e-117 |
| M6 delta Hurst | 0.491 | 0.140 | 0.036 | -0.313 [-0.390, -0.240] | 1.4e-121 |
| M6 box dimension | 1.892 | 2.242 | 2.346 | 2.695 [2.687, 2.703] | 1.4e-121 |
| M7 texture amplitude | 0.279 | 0.583 | 0.625 | 1.656 [1.548, 1.778] | 8.7e-118 |
| M7 texture correspondence | 0.221 | 0.160 | 0.128 | 0.065 [0.061, 0.068] | 3.6e-118 |
