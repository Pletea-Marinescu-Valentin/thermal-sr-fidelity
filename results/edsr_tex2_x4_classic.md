# Texture-loss ablation (classic, x4)

n=1144 frames / 17 videos. Checkpoint runs/edsr_tex2_x4_classic/last.pt at step 15000.

| metric | Bicubic | EDSR (L1) | ESRGAN | EDSR + texture loss (lambda=0.002) | p vs EDSR (L1) |
|---|---|---|---|---|---|
| PSNR (dB) | 50.264 | 53.169 | 50.964 | 51.913 [50.174, 53.739] | 1.2e-04 |
| LPIPS | 0.094 | 0.033 | 0.015 | 0.028 [0.020, 0.036] | 1.2e-04 |
| M0 RMSE (K) | 0.591 | 0.474 | 0.580 | 0.519 [0.404, 0.678] | 1.2e-04 |
| M4 texture ratio | 0.434 | 0.423 | 1.113 | 1.482 [1.396, 1.595] | 1.2e-04 |
| M6 delta Hurst | 0.766 | 0.744 | -0.044 | -0.121 [-0.149, -0.099] | 1.2e-04 |
| M6 box dimension | 1.759 | 1.781 | 2.569 | 2.645 [2.558, 2.721] | 1.2e-04 |
| M7 texture amplitude | 0.323 | 0.279 | 1.141 | 1.811 [1.641, 2.020] | 1.2e-04 |
| M7 texture correspondence | 0.143 | 0.173 | 0.039 | 0.030 [0.025, 0.035] | 1.2e-04 |
