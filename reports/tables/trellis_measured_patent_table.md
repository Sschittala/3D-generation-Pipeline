# Patent-level TRELLIS evaluation

| Metric | Mean ± SD | Patents |
| --- | ---: | ---: |
| SSIM | 0.6506 ± 0.1392 | 50 |
| PSNR (dB) | 11.9105 ± 2.6989 | 50 |
| LPIPS | 0.5316 ± 0.1286 | 50 |
| CLIP image similarity | 0.7130 ± 0.0381 | 50 |
| CLIP caption similarity | 0.2408 ± 0.0197 | 50 |

Patent-level evaluation of imported TRELLIS reconstructions on 438 patent drawings from 50 patents. Scores are averaged over eight rendered viewpoints per drawing, then over drawings within each patent. Values report the mean and sample standard deviation across patents. CLIP image similarity compares drawings with renders; CLIP caption similarity compares patent captions with renders. These image-based scores do not directly measure 3D geometric accuracy.

Source: data/metrics/TRELLIS_Complete_Results/metrics_summary.csv. No adjusted New Pipeline or Know3D columns are used. Silhouette IoU and contour Chamfer are excluded pending foreground-mask validation. The eight-view setting is supported by the imported per-view archive; original run settings and generation provenance should still be documented.
