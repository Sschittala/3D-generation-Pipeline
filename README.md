# 3D Generation Pipeline

Pipeline scripts, evaluation metrics, and figures for the TRELLIS, text-guided,
and SAP-Know3D experiments.

## Project layout

```text
.
├── README.md
├── SketchDream/         # Separate upstream Git repository
├── TRELLIS/             # Separate upstream Git repository
├── scripts/             # Pipeline, evaluation, analysis, and plotting scripts
├── data/
│   └── metrics/         # Evaluation CSVs
└── reports/
    └── figures/         # Generated PNG charts
```

## Running scripts

From the project root, for example:

```bash
python scripts/check_sim.py
python scripts/least_matched.py
python scripts/generate_graph_3pipelines.py
```

Module execution also works, for example `python -m scripts.check_sim`.
Local CSV and figure paths resolve relative to the project, regardless of the
current working directory. Script filenames have been preserved.

| Purpose | Scripts |
| --- | --- |
| Pipeline and evaluation | `run_vista_pipeline.py`, `sap_know3d_production.py`, `run_know3d_batch_eval.py`, `textverse_pipeline.py` |
| Metrics inspection | `baseline.py`, `check_sim.py`, `least_matched.py`, `check_path.py` |
| Charts | `graph.py`, `generate_grphs.py`, `generate_graph_3pipelines.py` |
| Synthetic example data | `generate_texverse_csv.py` |

Scripts require their existing Python dependencies (such as pandas, NumPy,
matplotlib, seaborn, PyTorch, OpenCV, Pillow, or huggingface_hub, depending on
the script). See each upstream repository for its own setup instructions.

## Data and external inputs

The three existing evaluation CSVs are in `data/metrics/`. Generated charts go
to `reports/figures/`; existing figures, including numbered variants, are retained.
Scripts that generate metrics or charts can overwrite their corresponding outputs.

`baseline.py`, `graph.py`, `generate_grphs.py`, and `check_path.py` still reference
external inputs under `/work/11649/schittala/vista/`. Those inputs are not included
here. `run_vista_pipeline.py` first checks `data/metrics/metrics_summary.csv`, then
searches its configured Vista workspace. `sap_know3d_production.py` optionally
reads `data/metrics/metrics_per_view.csv`.

The existing TexVerse CSV generator uses random synthetic scores, and the Vista
and Know3D evaluation scripts adjust existing scores. These outputs should be
interpreted according to those implementations.

## Sketch-to-3D comparison

See [experiment setup and current limitations](experiments/sketch_to_3d/README.md)
for the imported TRELLIS audit and new staged experiment runner.
