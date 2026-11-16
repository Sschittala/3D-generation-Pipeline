# Additional qualitative examples

Selection was fixed before viewing new images: rank all 50 patents by patent-mean recorded LPIPS, choose minimum, quartile, median, third-quartile and maximum rank, then choose the drawing nearest each selected patent mean. This spreads examples across one recorded metric; it does not establish human-rated typicality or genuine failure. No patent-level quality ordering beyond LPIPS is claimed.

Run on the Mac after authenticating as needed. These commands copy saved evaluation folders, not new generations:

```bash
scp -r schittala@vista.tacc.utexas.edu:/work/11649/schittala/vista/impact_3d_outputs/D0918440/evaluation/USD0918440-20210504-D00006 /Users/sai/3D-generation-Pipeline-1/experiments/sketch_to_3d/D0918440_USD0918440-20210504-D00006_evaluation
scp -r schittala@vista.tacc.utexas.edu:/work/11649/schittala/vista/impact_3d_outputs/D0913942/evaluation/USD0913942-20210323-D00001 /Users/sai/3D-generation-Pipeline-1/experiments/sketch_to_3d/D0913942_USD0913942-20210323-D00001_evaluation
scp -r schittala@vista.tacc.utexas.edu:/work/11649/schittala/vista/impact_3d_outputs/D0913174/evaluation/USD0913174-20210316-D00005 /Users/sai/3D-generation-Pipeline-1/experiments/sketch_to_3d/D0913174_USD0913174-20210316-D00005_evaluation
scp -r schittala@vista.tacc.utexas.edu:/work/11649/schittala/vista/impact_3d_outputs/D0915081/evaluation/USD0915081-20210406-D00000 /Users/sai/3D-generation-Pipeline-1/experiments/sketch_to_3d/D0915081_USD0915081-20210406-D00000_evaluation
scp -r schittala@vista.tacc.utexas.edu:/work/11649/schittala/vista/impact_3d_outputs/D0937859/evaluation/USD0937859-20211207-D00000 /Users/sai/3D-generation-Pipeline-1/experiments/sketch_to_3d/D0937859_USD0937859-20211207-D00000_evaluation
```

Original TIFF validation is separately pending. Copy originals using the exact tiff_path entries in metrics_summary.csv; do not infer original masks from saved resized reference images. Once copied, inspect all chosen examples and retain failed evaluations rather than silently replacing them.
