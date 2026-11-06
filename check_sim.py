import pandas as pd

# Load the new pipeline metrics
df = pd.read_csv('metrics_new_pipeline.csv')

# Sort by combined text and image similarity to find the best overall matching samples
df['combined_match_score'] = (df['best_clip_caption_similarity'] + df['best_clip_image_similarity']) / 2
top_matches = df.sort_values(by='combined_match_score', ascending=False).head(10)

print("=== Top 10 Best-Matching Patent Samples (Text & Image Alignment) ===")
print(top_matches[['patent_id', 'filename', 'best_clip_image_similarity', 'best_clip_caption_similarity', 'combined_match_score']].to_markdown(index=False))
