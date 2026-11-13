# Sketch-to-3D experiment status

The imported TRELLIS outputs are available, but the original generation script,
patent sketches, captions/prompts, and evaluation code were not found locally.
An ignored-file search includes `scripts/`, `TRELLIS/`, and virtual-environment
files outside installed third-party packages. `trellis-env` is a dependency
environment, not an experiment archive; its Python executable points to a
missing machine-specific Python installation on this Mac.

## Implemented here

- `audit.py`: standard-library audit of imported models, row counts, missing
  generation logs, and recorded metrics. Run `python3 experiments/sketch_to_3d/audit.py`.
- `imported_results_audit.json`: audit of all 438 imported GLBs. This validates
  headers/JSON and presence, not complete glTF compliance or visual correctness.
- `run.py image`: sketch + prompt to image through lineart ControlNet.
- `run.py trellis`: image to GLB, or sketch directly to GLB as an image-only baseline.
- Each actual run records input SHA-256, model identifiers, package versions,
  GPU name, seed, stage timings, memory, and optional estimated active cost.
  Use immutable local checkpoint paths if exact model revisions are required.

These are unvalidated GPU inference entry points. Only audit, argument validation,
syntax, and dry-run plans have been exercised on this machine. No new quality or
speed results have been generated. No model weights were downloaded.

## Environments

Use a compatible Linux NVIDIA CUDA machine and the setup in `TRELLIS/README.md`
for the TRELLIS stage. The image stage needs PyTorch with CUDA, Pillow, diffusers,
transformers, accelerate, and safetensors; use a separate environment if dependency
versions conflict. The two stages run in separate processes to release model memory.
Record resolved dependency versions for the environments actually used.

The image implementation follows the official Diffusers ControlNet interface:
https://huggingface.co/docs/diffusers/api/pipelines/controlnet

## Two-stage route

From the repository root, with a real sketch and its actual caption:

```bash
python experiments/sketch_to_3d/run.py image \
  --input /absolute/path/sketch.png \
  --prompt 'A technical render of the object described in the patent, on a plain white background' \
  --output experiments/sketch_to_3d/runs/sample/image --seed 42

python experiments/sketch_to_3d/run.py trellis \
  --input experiments/sketch_to_3d/runs/sample/image/image.png \
  --output experiments/sketch_to_3d/runs/sample/mesh --seed 42
```

Replace the example prompt with an object-specific description. The lineart input
must use dark lines on a light background. Inputs are padded to 512x512 without
changing aspect ratio. Inspect `control_input.png` and `image.png` before 3D conversion.
Use `--dry-run` with an existing input to validate a plan without model imports.
Outputs require a new directory so prior runs cannot be silently overwritten.

Text conditions the **image stage only**. Stock TRELLIS image-to-3D does not accept
text; the runner rejects `--prompt` for that stage. This route does not satisfy
independent text conditioning at both stages. That requires a different model or
an explicitly implemented and validated conditioning mechanism.

## Direct sketch-input baseline

```bash
python experiments/sketch_to_3d/run.py trellis \
  --input /absolute/path/sketch.png \
  --output experiments/sketch_to_3d/runs/sample/direct --seed 42
```

This tests TRELLIS treating the sketch as its image input. It is not text-guided,
and the imported results cannot be assigned to this route without their source code.

## SketchDream: sketch + text to 3D

The existing `SketchDream/` repository implements this route, with internal depth,
multiview diffusion, and iterative 3D optimization. It is not a single inference step.
Its `models/` directory is missing here. Obtain its documented checkpoints and use
its own compatible CUDA environment. A provided example can be launched from
`SketchDream/` using the correct matching config and both text processors:

```bash
python launch.py --config configs/tank.yaml --train --gpu 0 \
  system.prompt_processor.prompt='a 3D model of realistic camouflage tank' \
  system.lucid_prompt_processor.text='a 3D model of realistic camouflage tank'
```

For patent inputs, provide a corresponding sketch, RGBA mask, depth array, and
matching camera settings via a dedicated config. `tools/depth_predict.py` and
`scripts/depth_predict.sh` provide the depth-generation interface and require
additional model files. Include that preprocessing time, any retries, the full
optimization run, and mesh export in the comparison. SketchDream has not been
run or instrumented by the new runner.

## Fair efficiency and cost comparison

Use the same patent inputs, object-specific prompts, seeds, output resolution,
export settings, hardware, and quality checks. Begin with a small matched subset.
Keep measured results separate from the existing synthetic score-adjustment scripts.

Report setup/model-loading, preprocessing, inference/optimization, export, and
total time separately. Run repeated seeds and distinguish cold starts from warm
runs. Record failures and retries. Include silhouette alignment and mask checks,
mesh fragmentation/watertightness, and original-sketch/render comparisons.
A caption similarity score is not evidence that text conditioned generation.

Pass `--gpu-hourly-rate RATE` for a user-supplied USD/hour rate. Estimated active
cost is `total_seconds / 3600 * RATE`; it excludes idle allocation, minimum billing,
storage, and transfer. Sum both stages for the two-stage route. The imported
6.09-second average lacks timing-scope metadata and is not yet directly comparable.

## Still required

The Vista generation and evaluation scripts, input patent TIFFs/images, actual
prompts, model/settings provenance, GPU type, and billing rate. Pulling Git changes
only transfers tracked files; server-local datasets, untracked job scripts, model
caches, and ignored files may require a separate copy. The CSVs reference source
inputs under `/work/11649/schittala/vista/IMPACT/Sample data/` and outputs under
`/work/11649/schittala/vista/impact_3d_outputs/`.
