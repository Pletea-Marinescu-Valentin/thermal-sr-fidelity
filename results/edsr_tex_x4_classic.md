# Texture-loss ablation (classic, x4)

n=1144 frames / 17 videos. Checkpoint runs/edsr_tex_x4_classic/last.pt at step 15000.

| metric | Bicubic | EDSR (L1) | ESRGAN | EDSR + texture loss | p vs EDSR (L1) |
|---|---|---|---|---|---|
| PSNR (dB) | 50.264 | 53.169 | 50.964 | 51.615 [49.782, 53.553] | 1.2e-04 |
| LPIPS | 0.094 | 0.033 | 0.015 | 0.026 [0.019, 0.033] | 1.2e-04 |
| M0 RMSE (K) | 0.591 | 0.474 | 0.580 | 0.538 [0.417, 0.702] | 1.2e-04 |
| M4 texture ratio | 0.434 | 0.423 | 1.113 | 1.407 [1.309, 1.536] | 1.2e-04 |
| M6 delta Hurst | 0.766 | 0.744 | -0.044 | -0.142 [-0.184, -0.110] | 1.2e-04 |
| M6 box dimension | 1.759 | 1.781 | 2.569 | 2.666 [2.596, 2.732] | 1.2e-04 |
| M7 texture amplitude | 0.323 | 0.279 | 1.141 | 1.683 [1.493, 1.924] | 1.2e-04 |
| M7 texture correspondence | 0.143 | 0.173 | 0.039 | 0.032 [0.027, 0.037] | 1.2e-04 |
