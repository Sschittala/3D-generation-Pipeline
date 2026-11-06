"""Shared paths, resolved independently of the current working directory."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
METRICS_DIR = PROJECT_ROOT / "data" / "metrics"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
