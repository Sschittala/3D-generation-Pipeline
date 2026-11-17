# Sketch-input adaptation ablation for TRELLIS

This implements a new experiment, not a claimed improvement. No GPU generation or new reconstruction scores have been produced locally. The existing imported outputs, adjusted scores, and synthetic TexVerse CSVs are not overwritten or reused as ablation results.

## Configurations

| Variant | Common reviewed crop/canvas | TRELLIS background preprocessing | Input representation |
| --- | --- | --- | --- |
| stock | Yes | Enabled | Original RGB crop |
| bypass | Yes | Disabled | Same RGB crop |
| distance | Yes | Disabled | Black soft distance-to-ink map on white |
| distance_blur | Yes | Disabled | Same map followed by Gaussian smoothing |

The common canvas preserves aspect ratio and defaults to 518 pixels square. Thin strokes are retained by Otsu thresholding without morphological opening or speck removal. Distance is measured in canvas pixels; map strength is `exp(-0.5*(distance/width)^2)`. This is not a boundary probability or a filled object mask. Defaults `width=2` and `blur=1` are provisional settings for development, not validated optimum values. Both stock and bypass see identical pixels; only stock calls TRELLIS with `preprocess_image=True`. The image model receives no text or camera conditioning.

The fixed input size, export settings, seeds and model are shared across variants. Use the same immutable local checkpoint directory for exact model provenance. A remote model ID can change; the runner records the ID, TRELLIS commit and pipeline-source hash but does not pin or fully hash a remote checkpoint. Use the existing, working Vista TRELLIS environment; no environment or GPU setup is modified by these scripts.

## 1. Copy code from the Mac

Run in the Mac terminal:

```bash
scp -r /Users/sai/3D-generation-Pipeline-1/experiments/sketch_adaptation schittala@vista.tacc.utexas.edu:/work/11649/schittala/vista/
ssh schittala@vista.tacc.utexas.edu
```

All subsequent commands below run on Vista. Activate the same Python environment that successfully ran `generate_all_impact_3d.py`. Its activation path is not known locally. Do not install a new PyTorch or CUDA stack over that working environment.

```bash
cd /work/11649/schittala/vista
export SPCONV_ALGO=native
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export PYOPENGL_PLATFORM=egl
python -c 'import torch, cv2, PIL; print(torch.__version__)'
```

## 2. Create and review a development manifest

Start with one known patent for a smoke test. This does not provide patent-level confidence intervals or publication evidence.

```bash
python sketch_adaptation/make_manifest.py \
  --impact-dir /work/11649/schittala/vista/IMPACT \
  --patent-ids D0908314 \
  --split dev \
  --output /work/11649/schittala/vista/sketch_adaptation/pilot_candidates.jsonl
```

Each JSON line contains a source path, unique sample ID, patent ID, caption, split and crop. `crop` is `[left, top, right, bottom]` in source-image pixels, with right/bottom exclusive. The starter boxes cover the entire image and `crop_reviewed` is false. Inspect the actual TIFFs, edit the boxes to isolate one figure without labels, and set `crop_reviewed` to true only after inspection. Do not change captions separately for drawings of the same patent: the original evaluator expects one patent-level caption.

For a quick one-drawing pilot, retain the first entry and edit that file:

```bash
head -n 1 sketch_adaptation/pilot_candidates.jsonl > sketch_adaptation/pilot.jsonl
nano sketch_adaptation/pilot.jsonl
```

To create multiple crops from one TIFF, duplicate its entry with a different sample ID and crop; retain the same patent and split. Review whether the patent-level caption accurately describes each figure; caption scoring can be inappropriate for partial/section views. Missing captions remain missing, not zero.

## 3. Prepare the inputs on a compute node

Request an interactive GPU allocation before generation. TACC documents this form for Vista:

```bash
idev -p gh-dev -N 1 -n 1 -t 1:00:00
```

If an allocation account is required, supply your actual account with `-A YOUR_ACCOUNT`. Reactivate the working TRELLIS environment after entering the node, if needed, and set the environment variables above. Check that CUDA is visible:

```bash
python -c 'import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))'
cd /work/11649/schittala/vista
python sketch_adaptation/run.py prepare \
  --manifest sketch_adaptation/pilot.jsonl \
  --output sketch_adaptation/prepared_pilot_v1 \
  --size 518 --width 2 --blur 1
```

Inspect `prepared_pilot_v1/inputs/<sample_id>/crop.png`, `reference.png`, `ink_mask.png`, `distance.png` and `distance_blur.png` before inference. Copy them to your Mac if necessary. Preparation refuses unreviewed crops. It does not automatically identify individual figures or correct image polarity. Dark-background or blank images need inspection; the method assumes dark ink on a light background.

Each preparation uses a new output directory and saves source/input hashes and protocol parameters. Altering crops or map parameters requires a new preparation. Do not tune them using held-out test results.

## 4. Plan and generate the four matched variants

The dry run checks paths and prints the plan without importing TRELLIS or loading weights:

```bash
python sketch_adaptation/run.py generate \
  --prepared sketch_adaptation/prepared_pilot_v1 \
  --output sketch_adaptation/runs/pilot_v1 \
  --trellis-root /work/11649/schittala/vista/TRELLIS \
  --split dev --seeds 1 --dry-run
```

Run the same command without `--dry-run`:

```bash
python sketch_adaptation/run.py generate \
  --prepared sketch_adaptation/prepared_pilot_v1 \
  --output sketch_adaptation/runs/pilot_v1 \
  --trellis-root /work/11649/schittala/vista/TRELLIS \
  --split dev --seeds 1
```

All four variants run by default. For a larger experiment use `--seeds 1 2 3`. `--model /absolute/path/to/checkpoint` selects a fixed local checkpoint. Use a fresh run directory for every invocation: there is intentionally no silent resume or overwrite. On per-sample errors the runner records failures, continues other samples and exits nonzero at the end. Do not relabel or omit failed cases. Inspect failures before proceeding to evaluation; a run with errors can still be evaluated and summarized explicitly.

Outputs are `runs/<run>/<variant>/seed-<seed>/<patent>/<sample>.glb` plus individual `.run.json` files and a run-level provenance record. The pipeline timer excludes preparation, model loading and export; separate setup/export times are recorded. It is not end-to-end deployment latency. Peak allocated memory includes model allocations. No generation PLY is required for this ablation; the GLB exporter uses the generated Gaussian representation.

## 5. Evaluate against the SAME reference for all variants

The prepared `IMPACT/Sample data` reference dataset contains the untransformed reviewed RGB crops, never the distance maps. The wrapper calls the copied/original evaluator in a separate process for each variant and seed. The existing image metrics are retained for comparability; its silhouette-mask limitations are not fixed by input adaptation.

```bash
python sketch_adaptation/results.py evaluate \
  --run sketch_adaptation/runs/pilot_v1 \
  --evaluator /work/11649/schittala/vista/TRELLIS/evaluate_trellis_outputs.py

python sketch_adaptation/results.py summarize \
  --run sketch_adaptation/runs/pilot_v1
```

Use `evaluate --dry-run` to print commands first. Evaluation records its command and evaluator hash and refuses to overwrite existing summary CSVs. It checks reference hashes. Run preparation/evaluation in the working environment with OpenCV, Pillow, numpy, torch, pyrender/EGL, trimesh, LPIPS, scipy, scikit-image and transformers already available, as used by the original evaluator. GPU job time must cover export and evaluation as well as inference; do not assume the historical pipeline-only timing predicts total duration.

Results appear in `runs/pilot_v1/analysis/results.md`, `paired_metrics.csv`, and `coverage.csv`. Each metric uses the common complete-case image set across ALL requested variants/seeds. Seeds are averaged within images, images within patents, then patents weighted equally. The table reports mean ± patent SD and paired changes relative to stock, with a fixed-seed 2,000-resample patent bootstrap. A one-patent pilot has no uncertainty interval. Coverage includes planned, generated, evaluated and missing/failed counts; report coverage alongside the complete-case scores. Intervals are exploratory and not corrected for multiple comparisons. Lower LPIPS is better; higher SSIM/PSNR/CLIP is better. Scores are not geometry-ground-truth accuracy.

Copy the results back on the Mac:

```bash
scp -r schittala@vista.tacc.utexas.edu:/work/11649/schittala/vista/sketch_adaptation/runs/pilot_v1/analysis /Users/sai/3D-generation-Pipeline-1/reports/sketch_adaptation_pilot_v1
```

## 6. Development versus held-out test

Select development and test patents before tuning. Use `make_manifest.py --split dev` for development patents and `--split test` for disjoint test patents; combine the reviewed JSONL records into one manifest. Preparation rejects a patent appearing in both splits. If preparing splits in different directories, manually preserve this separation: validation cannot detect other unrelated preparations.

Freeze crop rules, map width, blur, seed list, model revision and evaluation protocol using development data. Then prepare the test inputs with the SAME parameters and invoke generation with `--split test` and a new run directory. Use enough distinct patents to characterize variation, and report the selection and exclusions. Do not choose the best seeds or favorable patents after viewing results.

A baseline on manually cropped images is not equivalent to the historical whole-TIFF baseline. For a claim about cropping itself, add a separately defined whole-TIFF experiment; the four matched variants here isolate preprocessing on a shared reviewed crop. Benefits from crops cannot be attributed to the distance map.

## Batch execution

`vista_generate.sbatch` generates the four variants after you activate the working environment and export its absolute Python path. Supply the preparation and output paths as arguments; customize allocation account and time limit at submission. It does not run evaluation automatically, so inspect generation logs and resource use before scheduling evaluation. The same evaluation command can run in an interactive GPU allocation.

For example, from `/work/11649/schittala/vista` after activating the working environment (replace YOUR_ACCOUNT):

```bash
export TRELLIS_PYTHON="$(command -v python)"
sbatch -A YOUR_ACCOUNT sketch_adaptation/vista_generate.sbatch \
  /work/11649/schittala/vista/sketch_adaptation/prepared_pilot_v1 \
  /work/11649/schittala/vista/sketch_adaptation/runs/three_seed_v1 dev
```

Unlike the one-seed interactive pilot, this template runs three seeds. Submit a new run directory, and adjust the job time after measuring the pilot.

## CPU verification

From the local project root in an environment with Pillow, numpy and OpenCV:

```bash
python -m unittest discover -s tests -p test_sketch_adaptation.py
```

Checks cover thin-line preservation, blank-input rejection, patent split leakage and equal-patent paired summaries with missing cases. GPU inference, export, and the original evaluator require validation on Vista.

Scheduler reference: https://docs.tacc.utexas.edu/hpc/vista/ (single-node PyTorch / idev example).

## Optional attention fallback: no FlashAttention compilation

If neither external FlashAttention nor xformers is available, `run.py generate --attention-backend sdpa` opts into native PyTorch scaled-dot-product attention. The runner installs a process-local adapter for the six noncausal, zero-dropout FlashAttention entry points used by the inspected TRELLIS source. Dense attention uses TRELLIS's SDPA branch; sparse attention uses the adapter, with variable-length sequences bucketed separately to preserve sample isolation. No TRELLIS source or installed packages are modified. This is intentionally not a general FlashAttention implementation; unsupported extra arguments fail explicitly.

Before model generation, run the included numerical self-test in the actual Vista environment:

```bash
python3 sketch_adaptation/sdpa_compat.py --device cuda
```

It compares all six calls against explicit attention calculations and tests unequal sequence lengths and isolation. Run generation only if it reports PASS. This self-test and full CUDA inference were not executable locally because PyTorch is unavailable in this workspace. Local checks cover syntax, adapter registration and offset validation; GPU performance and numerical behavior remain to be verified on Vista.

Then add `--attention-backend sdpa` to generation and choose a new output directory. For example:

```bash
python3 sketch_adaptation/run.py generate \
  --prepared sketch_adaptation/prepared_pilot_v1 \
  --output sketch_adaptation/runs/pilot_sdpa_v1 \
  --trellis-root /work/11649/schittala/vista/TRELLIS \
  --split dev --seeds 1 --attention-backend sdpa
```

The log still prints sparse attention as `flash_attn` because that is the routing entry point in TRELLIS, but the runner explicitly announces the adapter override and records its hash. Actual computation is PyTorch SDPA. Use the same backend for every ablation. Memory use and speed may differ from external FlashAttention, so do not combine resource comparisons across backends. Native SDPA may select a fused CUDA kernel or a slower mathematical kernel depending on device, dtype and dimensions. A short pilot is required; success is not guaranteed for an uninspected server-side TRELLIS revision.

## Unattended pilot when interactive queues are busy

`vista_pilot.sbatch` runs the one-seed/four-configuration pilot on `gh`, then evaluates and summarizes it in the same allocation. It restores modules and the existing environment, checks dependencies and CUDA, tests external FlashAttention, and explicitly selects the native SDPA adapter for the entire experiment if the external backend fails. It never installs packages. The configured allocation is TG-NAIRR25008, taken from the user's successful idev submission; change it if Slurm rejects it. The queue may still have a wait; batch submission does not promise faster scheduling.

Copy the updated run.py, results.py, sdpa_compat.py and vista_pilot.sbatch from the Mac to the server's sketch_adaptation directory, then submit on a Vista login node:

```bash
cd /work/11649/schittala/vista
sbatch sketch_adaptation/vista_pilot.sbatch
squeue -u schittala -o "%.18i %.12P %.10T %.10M %.10l %R"
```

Output is `sketch-pilot-JOBID.out` and `sketch-pilot-JOBID.err` in the submission directory; results are in `sketch_adaptation/runs/batch-pilot-JOBID/analysis/`. You can disconnect after successful submission. Use `tail -n 60 sketch-pilot-JOBID.out` to inspect progress, and `sacct -j JOBID --format=JobID,State,Elapsed,ExitCode` when a job leaves squeue. Job-specific output directories avoid collisions with failed interactive runs. Do not submit a second pilot while this request is pending.
