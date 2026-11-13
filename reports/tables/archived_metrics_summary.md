# Archived metric summaries — not a measured pipeline comparison

These tables describe the checked-in CSVs. Do not present them as independent experimental results.
The New Pipeline generator randomly adjusts baseline scores; the Know3D generator copies those rows and scales best-view CLIP scores; the TexVerse generator samples random distributions. Generation provenance for the archived meshes remains unverified.

Patent archive: New Pipeline 438 images / 50 patents; Know3D 438 images / 50 patents. Matched image identifiers: 438. Shared quality metrics identical within 1e-12 tolerance: True.

Cells report mean ± sample standard deviation; n is the number of valid aggregation units. Blank and non-finite values are excluded, never replaced by zero. A single observation has SD=NA.
Patent-level results first average valid image-level mean_* scores within each patent, then weight patents equally. Image-level results weight images equally. Neither SD measures uncertainty across generation seeds. No significance tests are warranted by these adjusted/synthetic archives.
Missing contour values may bias summaries; report their counts. Best-view columns and scaled Know3D CLIP columns are excluded from the quality tables. TexVerse sample IDs have no verified mapping to patents, so the datasets are not pooled.

## Patent-level summary

| Metric | New Pipeline | Know3D |
| --- | --- | --- |
| SSIM ↑ | 0.6688 ± 0.1422 (n=50) | 0.6688 ± 0.1422 (n=50) |
| PSNR (dB) ↑ | 11.9105 ± 2.6989 (n=50) | 11.9105 ± 2.6989 (n=50) |
| LPIPS ↓ | 0.5316 ± 0.1286 (n=50) | 0.5316 ± 0.1286 (n=50) |
| Silhouette IoU ↑ | 0.0173 ± 0.0188 (n=50) | 0.0173 ± 0.0188 (n=50) |
| Normalized contour Chamfer ↓ | 0.0890 ± 0.0327 (n=47) | 0.0890 ± 0.0327 (n=47) |
| CLIP image similarity ↑ | 0.7403 ± 0.0411 (n=50) | 0.7403 ± 0.0411 (n=50) |
| CLIP caption similarity ↑ | 0.3575 ± 0.0204 (n=50) | 0.3575 ± 0.0204 (n=50) |

## Image-level summary

| Metric | New Pipeline | Know3D |
| --- | --- | --- |
| SSIM ↑ | 0.7166 ± 0.1509 (n=438) | 0.7166 ± 0.1509 (n=438) |
| PSNR (dB) ↑ | 12.7109 ± 4.1520 (n=438) | 12.7109 ± 4.1520 (n=438) |
| LPIPS ↓ | 0.4826 ± 0.1425 (n=438) | 0.4826 ± 0.1425 (n=438) |
| Silhouette IoU ↑ | 0.0182 ± 0.0348 (n=438) | 0.0182 ± 0.0348 (n=438) |
| Normalized contour Chamfer ↓ | 0.0978 ± 0.0670 (n=398) | 0.0978 ± 0.0670 (n=398) |
| CLIP image similarity ↑ | 0.7332 ± 0.0663 (n=438) | 0.7332 ± 0.0663 (n=438) |
| CLIP caption similarity ↑ | 0.3529 ± 0.0335 (n=438) | 0.3529 ± 0.0335 (n=438) |

## TexVerse simulation only

| Simulated pipeline | CLIP image ↑ | CLIP caption ↑ | SSIM ↑ |
| --- | --- | --- | --- |
| Baseline TRELLIS | 0.5524 ± 0.0777 (n=438) | 0.2588 ± 0.0495 (n=438) | 0.6570 ± 0.0597 (n=438) |
| Text-guided | 0.6255 ± 0.0699 (n=438) | 0.3686 ± 0.0586 (n=438) | 0.7215 ± 0.0492 (n=438) |
| SAP-Know3D | 0.7184 ± 0.0596 (n=438) | 0.8107 ± 0.0415 (n=438) | 0.7589 ± 0.0398 (n=438) |

## To obtain a paper comparison

Collect the same original sketches, object-specific prompts, and independent outputs from all three pipelines. Use a common renderer, camera set, preprocessing, model versions, and metric implementations. Record failures and generation seeds. Preserve per-view measurements and aggregate views to images, then images to patents. Report paired differences with uncertainty resampled at the patent level once real independent results exist.

Required external inputs are documented in experiments/sketch_to_3d/README.md. This script performs no model inference, downloads, rendering, or new quality evaluation.

Sources: data/metrics/metrics_new_pipeline.csv; data/metrics/metrics_know3d_pipeline.csv; data/metrics/texverse_evaluation_results.csv. Provenance code: scripts/run_vista_pipeline.py; scripts/run_know3d_batch_eval.py; scripts/generate_texverse_csv.py.
