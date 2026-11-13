"""Summarize archived CSVs; does not run inference or establish measured performance."""
import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
METRICS = [
    ('mean_ssim', 'SSIM', 'up'),
    ('mean_psnr', 'PSNR (dB)', 'up'),
    ('mean_lpips', 'LPIPS', 'down'),
    ('mean_silhouette_iou', 'Silhouette IoU', 'up'),
    ('mean_contour_chamfer_norm', 'Normalized contour Chamfer', 'down'),
    ('mean_clip_image_similarity', 'CLIP image similarity', 'up'),
    ('mean_clip_caption_similarity', 'CLIP caption similarity', 'up'),
]


def number(value):
    if value in (None, ''):
        return None
    result = float(value)
    if not math.isfinite(result):
        return None
    return result


def summarize(rows, column, group=None):
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        value = number(row.get(column))
        if value is not None:
            groups[row[group] if group else i].append(value)
    values = [statistics.mean(items) for items in groups.values()]
    return {
        'n': len(values),
        'mean': statistics.mean(values) if values else None,
        'sd': statistics.stdev(values) if len(values) > 1 else None,
        'median': statistics.median(values) if values else None,
    }


def load(path, keys):
    with path.open(newline='') as source:
        reader = csv.DictReader(source)
        if not reader.fieldnames or not set(keys).issubset(reader.fieldnames):
            raise ValueError(f'{path}: missing identifier columns {keys}')
        rows = list(reader)
    identifiers = [tuple(row[key] for key in keys) for row in rows]
    if not rows or any(not all(key) for key in identifiers) or len(set(identifiers)) != len(rows):
        raise ValueError(f'{path}: empty data, blank identifiers, or duplicate identifiers')
    return rows


def cell(stats):
    if stats['mean'] is None:
        return 'Unavailable'
    sd = f"{stats['sd']:.4f}" if stats['sd'] is not None else 'NA'
    return f"{stats['mean']:.4f} ± {sd} (n={stats['n']})"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metrics-dir', type=Path, default=ROOT / 'data/metrics')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'reports/tables')
    args = parser.parse_args()
    new = load(args.metrics_dir / 'metrics_new_pipeline.csv', ['patent_id', 'filename'])
    know = load(args.metrics_dir / 'metrics_know3d_pipeline.csv', ['patent_id', 'filename'])
    tex = load(args.metrics_dir / 'texverse_evaluation_results.csv', ['sample_id'])
    index = lambda rows: {(r['patent_id'], r['filename']): r for r in rows}
    left, right = index(new), index(know)
    matched = set(left) & set(right)
    identical = len(left) == len(right) == len(matched) and all(
        (number(left[key].get(col)) is None and number(right[key].get(col)) is None)
        or (number(left[key].get(col)) is not None and number(right[key].get(col)) is not None
            and math.isclose(number(left[key][col]), number(right[key][col]), rel_tol=1e-12, abs_tol=1e-12))
        for key in matched for col, _, _ in METRICS)
    records = []
    sections = []
    for group, title in [('patent_id', 'Patent-level summary'), (None, 'Image-level summary')]:
        table = []
        for col, label, direction in METRICS:
            values = []
            for pipeline, rows in [('New Pipeline', new), ('Know3D', know)]:
                stats = summarize(rows, col, group)
                records.append(dict(dataset='patent_archive', pipeline=pipeline,
                                    aggregation='patent' if group else 'image', metric=col,
                                    provenance='adjusted_or_copied_archive', **stats))
                values.append(cell(stats))
            table.append([label + (' ↑' if direction == 'up' else ' ↓'), *values])
        sections.append((title, ['Metric', 'New Pipeline', 'Know3D'], table))
    table = []
    for prefix, label in [('baseline', 'Baseline TRELLIS'), ('text_guided', 'Text-guided'), ('know3d', 'SAP-Know3D')]:
        values = []
        for metric in ['clip_image', 'clip_caption', 'ssim']:
            stats = summarize(tex, f'{prefix}_{metric}')
            records.append(dict(dataset='texverse_simulation', pipeline=label, aggregation='sample',
                                metric=metric, provenance='synthetic', **stats))
            values.append(cell(stats))
        table.append([label, *values])
    sections.append(('TexVerse simulation only', ['Simulated pipeline', 'CLIP image ↑', 'CLIP caption ↑', 'SSIM ↑'], table))
    notes = [
        '# Archived metric summaries — not a measured pipeline comparison',
        '',
        'These tables describe the checked-in CSVs. Do not present them as independent experimental results.',
        'The New Pipeline generator randomly adjusts baseline scores; the Know3D generator copies those rows and scales best-view CLIP scores; the TexVerse generator samples random distributions. Generation provenance for the archived meshes remains unverified.',
        '',
        f'Patent archive: New Pipeline {len(new)} images / {len({r["patent_id"] for r in new})} patents; Know3D {len(know)} images / {len({r["patent_id"] for r in know})} patents. Matched image identifiers: {len(matched)}. Shared quality metrics identical within 1e-12 tolerance: {identical}.',
        '',
        'Cells report mean ± sample standard deviation; n is the number of valid aggregation units. Blank and non-finite values are excluded, never replaced by zero. A single observation has SD=NA.',
        'Patent-level results first average valid image-level mean_* scores within each patent, then weight patents equally. Image-level results weight images equally. Neither SD measures uncertainty across generation seeds. No significance tests are warranted by these adjusted/synthetic archives.',
        'Missing contour values may bias summaries; report their counts. Best-view columns and scaled Know3D CLIP columns are excluded from the quality tables. TexVerse sample IDs have no verified mapping to patents, so the datasets are not pooled.',
        '',
    ]
    latex = ['% DESCRIPTIVE ARCHIVE / SYNTHETIC DATA ONLY. Not measured comparative evidence.', '% Requires \\usepackage{booktabs}.']
    def escape(value):
        return value.replace('↑', r'$\uparrow$').replace('↓', r'$\downarrow$').replace('±', r'$\pm$').replace('_', r'\_')
    for title, headers, rows in sections:
        notes += [f'## {title}', '', '| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |']
        notes += ['| ' + ' | '.join(row) + ' |' for row in rows]
        notes.append('')
        latex += [r'\begin{table*}[t]', r'\centering', r'\small',
                  r'\caption{' + title + '. Descriptive archive only; adjusted or synthetic scores, not independently measured performance. Values are mean ' + r'$\pm$' + ' sample SD; n denotes valid units.}',
                  r'\begin{tabular}{l' + 'l' * (len(headers)-1) + '}', r'\toprule',
                  ' & '.join(map(escape, headers)) + r' \\', r'\midrule']
        latex += [' & '.join(map(escape, row)) + r' \\' for row in rows]
        latex += [r'\bottomrule', r'\end{tabular}', r'\end{table*}', '']
    notes += ['## To obtain a paper comparison', '',
              'Collect the same original sketches, object-specific prompts, and independent outputs from all three pipelines. Use a common renderer, camera set, preprocessing, model versions, and metric implementations. Record failures and generation seeds. Preserve per-view measurements and aggregate views to images, then images to patents. Report paired differences with uncertainty resampled at the patent level once real independent results exist.', '',
              'Required external inputs are documented in experiments/sketch_to_3d/README.md. This script performs no model inference, downloads, rendering, or new quality evaluation.', '',
              'Sources: data/metrics/metrics_new_pipeline.csv; data/metrics/metrics_know3d_pipeline.csv; data/metrics/texverse_evaluation_results.csv. Provenance code: scripts/run_vista_pipeline.py; scripts/run_know3d_batch_eval.py; scripts/generate_texverse_csv.py.', '']
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / 'archived_metrics_summary.md').write_text('\n'.join(notes))
    (args.output_dir / 'archived_metrics_summary.tex').write_text('\n'.join(latex))
    with (args.output_dir / 'archived_metrics_summary.csv').open('w', newline='') as target:
        writer = csv.DictWriter(target, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    print(f'Generated Markdown, LaTeX, and numeric CSV summaries in {args.output_dir}')


if __name__ == '__main__':
    main()
