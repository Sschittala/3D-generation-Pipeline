if __package__:
    from .project_paths import METRICS_DIR, FIGURES_DIR
else:
    from project_paths import METRICS_DIR, FIGURES_DIR

import os
import pandas as pd
import numpy as np
from PIL import Image
import cv2

class SAPKnow3DProductionPipeline:
    def __init__(self, metrics_per_view_path=METRICS_DIR / "metrics_per_view.csv", metrics_new_path=METRICS_DIR / "metrics_new_pipeline.csv"):
        print("Initializing SAP-Know3D Production Pipeline for Patent 3D Reconstruction...")
        self.df_views = pd.read_csv(metrics_per_view_path) if os.path.exists(metrics_per_view_path) else None
        self.df_new = pd.read_csv(metrics_new_path) if os.path.exists(metrics_new_path) else None

    def sketch_aware_preprocessing(self, tiff_path: str) -> np.ndarray:
        """Converts sparse patent line art into a dense structural boundary map."""
        img = cv2.imread(tiff_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return np.zeros((512, 512), dtype=np.uint8)
        inverted = cv2.bitwise_not(img)
        dist_transform = cv2.distanceTransform(inverted, cv2.DIST_L2, 5)
        normalized = cv2.normalize(dist_transform, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        dense_boundary_map = cv2.GaussianBlur(normalized, (15, 15), 0)
        return dense_boundary_map

    def get_view_specific_prompt(self, patent_id: str, filename: str, base_caption: str) -> str:
        """Maps orthographic views to azimuth angles for view-conditioned generation."""
        azimuth = 0.0
        if self.df_views is not None:
            match = self.df_views[(self.df_views['patent_id'] == patent_id) & (self.df_views['filename'] == filename)]
            if not match.empty:
                azimuth = float(match.iloc[0]['azimuth_deg'])
        return (
            f"A technical 3D CAD render of patent {patent_id}, viewed from azimuth angle {azimuth}°, "
            f"featuring precise structural boundaries, clean orthographic proportions, "
            f"and mechanical features described as: {base_caption}"
        )

    def run_evaluation_summary(self):
        if self.df_new is not None:
            print("\n=== SAP-Know3D Pipeline Evaluation Summary ===")
            print(f"Total Evaluated Samples: {len(self.df_new)}")
            print(f"Mean CLIP Caption Similarity: {self.df_new['best_clip_caption_similarity'].mean():.4f}")
            print(f"Mean CLIP Image Similarity:   {self.df_new['best_clip_image_similarity'].mean():.4f}")
            print(f"Mean Structural SSIM:         {self.df_new['best_ssim'].mean():.4f}")

if __name__ == "__main__":
    pipeline = SAPKnow3DProductionPipeline()
    pipeline.run_evaluation_summary()
