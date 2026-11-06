import os
# Let's search where metrics_summary.csv is actually located on the system / workspace
found_paths = []
for root, dirs, files in os.walk('/work/11649/schittala/vista'):
    if 'metrics_summary.csv' in files:
        found_paths.append(os.path.join(root, 'metrics_summary.csv'))
print("Found metrics_summary.csv at:", found_paths)


# Let's search where metrics_per_view.csv or impact_3d_outputs is located on TACC Vista workspace
for root, dirs, files in os.walk('/work/11649/schittala/vista'):
    if 'metrics_per_view.csv' in files:
        print(os.path.join(root, 'metrics_new_pipeline.csv'))
