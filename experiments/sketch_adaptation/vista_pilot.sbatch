#!/bin/bash -l
#SBATCH -J sketch-pilot
#SBATCH -A TG-NAIRR25008
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 01:00:00
#SBATCH -o sketch-pilot-%j.out
#SBATCH -e sketch-pilot-%j.err
set -eo pipefail
module load gcc cuda/12.6
module load python3/3.11.8
source /work/11649/schittala/vista/TRELLIS/trellis-env/bin/activate
set -u
cd /work/11649/schittala/vista
export SPCONV_ALGO=native
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export OPENCV_FOR_THREADS_NUM=1
export PYOPENGL_PLATFORM=egl
export PYTHONUNBUFFERED=1
export ATTN_BACKEND=flash_attn
export SPARSE_ATTN_BACKEND=flash_attn
PYTHON_BIN="$(command -v python3)"
RUN_DIR="/work/11649/schittala/vista/sketch_adaptation/runs/batch-pilot-${SLURM_JOB_ID}"
TASK_DIR=/work/11649/schittala/vista/sketch_adaptation
PREPARED_DIR="$TASK_DIR/prepared_pilot_v1"
EVALUATOR=/work/11649/schittala/vista/TRELLIS/evaluate_trellis_outputs.py
printf 'Job %s on %s\nPython: %s\nResults: %s\n' "$SLURM_JOB_ID" "$(hostname)" "$PYTHON_BIN" "$RUN_DIR"
module list

# Import-only checks catch environment issues before loading model weights.
"$PYTHON_BIN" - <<'PY'
import sys
import cv2
import torch
import PIL
import pyrender
import trimesh
import lpips
import transformers
import skimage
import scipy
print('Python executable:', sys.executable)
print('PyTorch:', torch.__version__, 'CUDA runtime:', torch.version.cuda)
assert torch.cuda.is_available(), 'No CUDA GPU visible'
print('GPU:', torch.cuda.get_device_name(0))
cv2.setNumThreads(1)
PY

# Pick one backend for the entire matched pilot, and record it in run.json.
if "$PYTHON_BIN" - <<'PY'
import torch
import flash_attn
print('External FlashAttention:', flash_attn.__version__, flash_attn.__file__)
qkv=torch.randn(1,16,3,2,64,device='cuda',dtype=torch.float16)
flash_attn.flash_attn_qkvpacked_func(qkv)
torch.cuda.synchronize()
print('External FlashAttention GPU smoke test passed')
PY
then
    ATTENTION=installed
else
    echo 'External FlashAttention unavailable or failed its smoke test; testing the explicit SDPA adapter.'
    "$PYTHON_BIN" "$TASK_DIR/sdpa_compat.py" --device cuda
    ATTENTION=sdpa
fi

echo "Starting four-branch, one-seed pilot with attention=$ATTENTION"
GENERATION_EXIT=0
"$PYTHON_BIN" "$TASK_DIR/run.py" generate \
    --prepared "$PREPARED_DIR" --output "$RUN_DIR" \
    --trellis-root /work/11649/schittala/vista/TRELLIS \
    --split dev --seeds 1 --attention-backend "$ATTENTION" || GENERATION_EXIT=$?

# Continue to coverage reporting after individual failures, but not setup failure.
"$PYTHON_BIN" - "$RUN_DIR" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1])/'run.json'
if not p.is_file(): raise SystemExit('Generation did not create a run record; inspect the job error log.')
config=json.loads(p.read_text())
if config.get('model_setup_status')!='success':
    raise SystemExit('Model setup failed; inspect run.json and the job error log.')
PY

EVALUATION_EXIT=0
"$PYTHON_BIN" "$TASK_DIR/results.py" evaluate \
    --run "$RUN_DIR" --evaluator "$EVALUATOR" --device cuda || EVALUATION_EXIT=$?
"$PYTHON_BIN" "$TASK_DIR/results.py" summarize --run "$RUN_DIR"
printf 'Results saved: %s/analysis/results.md\n' "$RUN_DIR"
if [ "$GENERATION_EXIT" -ne 0 ] || [ "$EVALUATION_EXIT" -ne 0 ]; then
    echo 'Pilot has generation or evaluation failures. Review coverage.csv and logs; results are partial.' >&2
    exit 1
fi
echo 'Pilot finished. Check coverage.csv for per-sample evaluation statuses before interpreting scores.'
