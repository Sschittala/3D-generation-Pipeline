if __package__:
    from .project_paths import METRICS_DIR, FIGURES_DIR
else:
    from project_paths import METRICS_DIR, FIGURES_DIR

# Let's generate a multi-pipeline comparative visualization script using texverse_evaluation_results.csv
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sns.set_theme(style="whitegrid")

# Load texverse evaluation results
df = pd.read_csv(METRICS_DIR / 'texverse_evaluation_results.csv')

# Compute means across the three pipelines for a comparative bar chart
metrics_names = ['CLIP Image Sim', 'CLIP Caption Sim', 'Structural SSIM']
baseline_means = [df['baseline_clip_image'].mean(), df['baseline_clip_caption'].mean(), df['baseline_ssim'].mean()]
text_guided_means = [df['text_guided_clip_image'].mean(), df['text_guided_clip_caption'].mean(), df['text_guided_ssim'].mean()]
know3d_means = [df['know3d_clip_image'].mean(), df['know3d_clip_caption'].mean(), df['know3d_ssim'].mean()]

x = np.arange(len(metrics_names))
width = 0.25

fig, ax = plt.subplots(figsize=(9, 6))
rects1 = ax.bar(x - width, baseline_means, width, label='Baseline TRELLIS', color='#b0c4de')
rects2 = ax.bar(x, text_guided_means, width, label='Text-Guided Pipeline', color='#f4a460')
rects3 = ax.bar(x + width, know3d_means, width, label='SAP-Know3D Adapted', color='#4169e1')

ax.set_ylabel('Score / Similarity', fontweight='bold', fontsize=12)
ax.set_title('Multi-Pipeline Performance Comparison on TexVerse Dataset (438 Samples)', fontweight='bold', fontsize=14, pad=15)
ax.set_xticks(x)
ax.set_xticklabels(metrics_names, fontweight='bold', fontsize=11)
ax.legend(loc='upper left', frameon=True)
ax.set_ylim(0, 1.0)

# Add value labels on bars
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)

plt.tight_layout()
plt.savefig(FIGURES_DIR / 'texverse_multi_pipeline_bar_comparison.png', dpi=300)
plt.close()
print("Successfully generated texverse_multi_pipeline_bar_comparison.png")
