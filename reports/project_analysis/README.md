# Project analysis outputs

Regenerate with `python3 scripts/build_project_analysis.py` (numpy, matplotlib).

## Tables

- dataset_summary.md / .tex: archive inventory and provenance.
- metric_completeness.md / .tex / .csv: valid and missing counts.
- mesh_quality.md / .tex: medians and quartiles from independently recomputed GLB topology.
- mesh_quality_per_model.csv: all recomputed model measurements.
- images_per_patent.csv: dataset sampling counts.

## Figures (PNG and SVG)

- metric_distributions: recorded imported metrics only; box plots with all observations. Boxes show Q1–Q3, center is median, whiskers extend to observations within 1.5 IQR. Points are deterministically jittered. Scores are not newly evaluated.
- images_per_patent: archive sampling distribution.
- qualitative_mesh_grid: deterministic minimum, median-ranked and maximum component-count meshes, each at three canonical azimuths. Untextured geometry views, not photometric evaluation renders. These are not evidence of sketch fidelity.
- topology_failure_candidates: fragmentation and open-boundary inspection candidates. Component colors repeat; disconnected parts may be intentional. Selection rules and exact filenames are in figure_selection.csv.

## Limitations and paper placement

Dataset inventory belongs with experimental setup. Recomputed topology can be reported as an imported-artifact analysis with index-connectivity definitions; native vertices are not welded. UV seams or duplicated positions can inflate index-connected component and boundary counts; these are not automatically physical cracks. Metric distributions and completeness belong in an archive audit appendix until original evaluation provenance is established. Synthetic/adjusted pipeline scores are excluded from these plots.

No source patent TIFFs are available at their recorded paths. Therefore an original-sketch/mesh comparison and verified missing-part, depth-error or silhouette-failure figure cannot be produced. The geometry grid and topology candidate figure are the available substitutes, explicitly labeled as inspection. To finish those comparisons, supply matched original sketches and camera/alignment information. Source assets under upstream dependency packages are unrelated to these patents.

Imported metric provenance is unverified; new-pipeline values were adjusted by scripts/run_vista_pipeline.py, Know3D copies/scales them, and TexVerse is simulated. Counts of processing success are not reconstruction-quality success rates. No new generation, image similarity evaluation, or significance testing was performed.

Sources: data/metrics/TRELLIS_Complete_Results/metrics_summary.csv, generation_metrics.csv and GLB files; data/metrics/metrics_new_pipeline.csv; metrics_know3d_pipeline.csv; texverse_evaluation_results.csv. Existing inputs and tables remain unchanged. LaTeX files require booktabs and have not been compiled against the paper template.

On this Mac, the installed plotting packages require the Intel interpreter without user-site packages:

```bash
PYTHONNOUSERSITE=1 MPLCONFIGDIR=/private/tmp/project-analysis-matplotlib arch -x86_64 /Library/Frameworks/Python.framework/Versions/3.13/bin/python3 scripts/build_project_analysis.py
```
