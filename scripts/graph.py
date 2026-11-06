if __package__:
    from .project_paths import METRICS_DIR, FIGURES_DIR
else:
    from project_paths import METRICS_DIR, FIGURES_DIR

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load datasets
df_base = pd.read_csv('/work/11649/schittala/vista/impact_3d_outputs/metrics_summary.csv')
df_new = pd.read_csv(METRICS_DIR / 'metrics_new_pipeline.csv')

# Set plotting style for CVPR paper quality
plt.figure(figsize=(12, 5))
sns.set_theme(style="whitegrid")

# Plot 1: CLIP Caption Similarity Distribution Shift
plt.subplot(1, 2, 1)
sns.kdeplot(df_base['best_clip_caption_similarity'], label='Baseline (TRELLIS)', fill=True, color='skyblue', alpha=0.5)
sns.kdeplot(df_new['best_clip_caption_similarity'], label='New Pipeline (Text-Guided)', fill=True, color='orange', alpha=0.5)
plt.title('Best CLIP Caption Similarity Distribution', fontsize=12, fontweight='bold')
plt.xlabel('Caption Similarity Score', fontsize=10)
plt.ylabel('Density', fontsize=10)
plt.legend(frameon=True)

# Plot 2: SSIM Structural Fidelity Distribution
plt.subplot(1, 2, 2)
sns.kdeplot(df_base['best_ssim'], label='Baseline (TRELLIS)', fill=True, color='skyblue', alpha=0.5)
sns.kdeplot(df_new['best_ssim'], label='New Pipeline (Text-Guided)', fill=True, color='orange', alpha=0.5)
plt.title('Best SSIM (Structural Fidelity) Distribution', fontsize=12, fontweight='bold')
plt.xlabel('SSIM Score', fontsize=10)
plt.ylabel('Density', fontsize=10)
plt.legend(frameon=True)

plt.tight_layout()
plt.savefig(FIGURES_DIR / 'cvpr_metrics_comparison.png', dpi=300)
print("Publication figure successfully generated: cvpr_metrics_comparison.png")
