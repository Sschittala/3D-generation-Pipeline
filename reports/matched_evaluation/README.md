# Matched evaluation inspection: D0908314

All five copied drawing folders are included, without quality-based sample exclusion. The reference images are the evaluator's saved aligned TIFF derivatives, not raw TIFFs. Each row shows the reference, the render selected by maximum recorded SSIM across eight views, and a fixed az090 render. This is one patent and one imported pipeline, not a representative three-method benchmark.

The mask diagnostic applies the evaluator's actual foreground function to the saved aligned reference. It reprocesses the raw render with the actual crop function. Because original TIFFs are absent, the reference mask is re-thresholded after resize and is not guaranteed to equal the mask used during the original evaluation. Diagnostic IoU is therefore separate from recorded IoU. Image checks include exact render-alignment comparison and recomputed SSIM on the saved images.

The thresholded patent foreground represents ink/shading rather than a filled object silhouette. Thin lines may be removed by 3x3 opening and component filtering; reference and shaded-render masks need visual validation before interpreting IoU or contour distances as object-shape accuracy. Independently cropping each image normalizes size and margins but does not register pose or orientation.

Suggested caption: Matched qualitative examples for patent D0908314. Left: preprocessed patent drawings. Middle: imported TRELLIS renders selected by highest SSIM over eight horizontal viewpoints. Right: the same meshes rendered at a fixed 90-degree azimuth. All five available drawing/model pairs from this patent are shown. Best-view selection is used for illustration and does not demonstrate camera correspondence or performance relative to other methods.

Use the comparison in qualitative results with its selection rule. Keep the mask diagnostic in evaluation validation or the appendix. Inspect examples from additional patents before making general quality claims. To assess filled silhouettes, validate object masks against manually checked references and use renderer depth/alpha for generated foreground in a separate, versioned evaluation; do not silently replace old scores.

Regenerate with scripts/build_matched_evaluation_figure.py (numpy, Pillow, OpenCV, matplotlib, scipy). Sources: copied D0908314_evaluation images, imported metrics_per_view.csv, and evaluate_trellis_outputs.py. No model inference or source-file changes were performed.

## Observed checks

All five selected-view SSIM values reproduce the recorded per-view scores to within 1.2e-16. Recreating alignment from each selected raw render exactly reproduces its saved aligned image. Re-thresholding the aligned reference yields an empty mask for four of five drawings; this is a diagnostic result, not a reconstruction of the original pre-resize mask. Some TIFF derivatives contain multiple patent figures and figure labels, and corresponding renders include duplicated object structures. The current crop step does not isolate a single figure. High full-image SSIM alone does not establish correct 3D shape or pose correspondence.
