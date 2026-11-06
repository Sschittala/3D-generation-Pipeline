if __package__:
    from .project_paths import METRICS_DIR, FIGURES_DIR
else:
    from project_paths import METRICS_DIR, FIGURES_DIR

import pandas as pd
import numpy as np

# Let's inspect the baseline and new pipeline CSVs to verify their distributions and check scientific validity
df_base = pd.read_csv('/work/11649/schittala/vista/impact_3d_outputs/metrics_summary.csv')
df_new = pd.read_csv(METRICS_DIR / 'metrics_new_pipeline.csv')

print("Baseline shape:", df_base.shape)
print("New pipeline shape:", df_new.shape)

# Verify summary statistics to make sure the data is consistent and check correlation or differences
print("\nBaseline Caption Sim Mean:", df_base['best_clip_caption_similarity'].mean())
print("New Pipeline Caption Sim Mean:", df_new['best_clip_caption_similarity'].mean())
