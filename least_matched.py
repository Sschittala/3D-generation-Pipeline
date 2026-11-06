import pandas as pd

# Load the new pipeline metrics
df = pd.read_csv('metrics_new_pipeline.csv')

# Calculate combined match score
df['combined_match_score'] = (df['best_clip_caption_similarity'] + df['best_clip_image_similarity']) / 2

# Sort in ascending order to find the worst/least-matched items
worst_matches = df.sort_values(by='combined_match_score', ascending=True).head(10)

print("=== Bottom 10 Least-Matched Patent Samples ===")
print(worst_matches[['patent_id', 'filename', 'best_clip_image_similarity', 'best_clip_caption_similarity', 'combined_match_score']].to_markdown(index=False))
