if __package__:
    from .project_paths import METRICS_DIR, FIGURES_DIR
else:
    from project_paths import METRICS_DIR, FIGURES_DIR

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set seaborn styling for publication quality
sns.set_theme(style="whitegrid")

print("Loading evaluation metrics datasets...")
# Load datasets from the workspace
df_view = pd.read_csv('/work/11649/schittala/vista/impact_3d_outputs/metrics_per_view.csv')
df_metrics = pd.read_csv(METRICS_DIR / 'metrics_new_pipeline.csv')

# ==========================================
# 1. Multi-View Azimuth Radar Chart
# ==========================================
print("Generating multi-view azimuth radar chart...")
view_avg = df_view.groupby('azimuth_deg')['clip_image_similarity'].mean().reset_index()

angles = np.linspace(0, 2 * np.pi, len(view_avg), endpoint=False).tolist()
scores = view_avg['clip_image_similarity'].tolist()
angles += angles[:1]
scores += scores[:1]

fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
ax.plot(angles, scores, color='orange', linewidth=2, linestyle='solid', label='Text-Guided Pipeline')
ax.fill(angles, scores, color='orange', alpha=0.3)

ax.set_theta_zero_location('N')
ax.set_theta_direction(-1)
ax.set_xticks(angles[:-1])
ax.set_xticklabels([f"{int(deg)}°" for deg in view_avg['azimuth_deg']])
ax.set_title('Multi-View CLIP Image Similarity Across Azimuth Angles', fontweight='bold', pad=20)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

plt.tight_layout()
plt.savefig(FIGURES_DIR / 'cvpr_radar_chart.png', dpi=300)
plt.close()
print("Saved: cvpr_radar_chart.png")

# ==========================================
# 2. Correlation Heatmap (Geometry vs. Text/Visual Alignment)
# ==========================================
print("Generating geometry vs. alignment correlation heatmap...")
subset_corr = df_metrics[['vertices', 'faces', 'best_ssim', 'best_clip_image_similarity', 'best_clip_caption_similarity']].corr()

plt.figure(figsize=(8, 6))
sns.heatmap(subset_corr, annot=True, cmap='Blues', fmt='.2f', linewidths=0.5, vmin=-1, vmax=1)
plt.title('Correlation Matrix: Geometry vs. Visual & Text Alignment', fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'cvpr_correlation_heatmap.png', dpi=300)
plt.close()
print("Saved: cvpr_correlation_heatmap.png")

print("All publication-ready visual assets successfully generated!")
