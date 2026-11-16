#!/bin/bash
#SBATCH -J sketch-ablation
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:00:00
#SBATCH -o sketch-ablation-%j.out
#SBATCH -e sketch-ablation-%j.err
set -euo pipefail
: "${TRELLIS_PYTHON:?Activate the working TRELLIS environment and export TRELLIS_PYTHON to its absolute python executable before sbatch}"
if [ "$#" -ne 3 ]; then
    echo 'Usage: sbatch -A YOUR_ACCOUNT vista_generate.sbatch PREPARED_DIR NEW_RUN_DIR dev_or_test' >&2
    exit 2
fi
export SPCONV_ALGO=native
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export PYOPENGL_PLATFORM=egl
TASK_DIR=/work/11649/schittala/vista/sketch_adaptation
"$TRELLIS_PYTHON" "$TASK_DIR/run.py" generate \
  --prepared "$1" --output "$2" --split "$3" \
  --trellis-root /work/11649/schittala/vista/TRELLIS \
  --seeds 1 2 3
