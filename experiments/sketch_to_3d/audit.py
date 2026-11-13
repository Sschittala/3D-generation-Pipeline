"""Audit imported TRELLIS artifacts without third-party Python dependencies."""
import csv
import json
import math
import statistics
import struct
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / 'data/metrics/TRELLIS_Complete_Results'


def read(name):
    with (RESULTS / name).open(newline='') as stream:
        return list(csv.DictReader(stream))


def mean(rows, key):
    values = [float(r[key]) for r in rows if r.get(key) and math.isfinite(float(r[key]))]
    return statistics.mean(values) if values else None


def main():
    summary, generation, views = read('metrics_summary.csv'), read('generation_metrics.csv'), read('metrics_per_view.csv')
    key = lambda row: (row['patent_id'], row['filename'])
    models = list(RESULTS.rglob('*.glb'))
    invalid = []
    for path in models:
        try:
            with path.open('rb') as stream:
                magic, version, length = struct.unpack('<4sII', stream.read(12))
                size, kind = struct.unpack('<II', stream.read(8))
                if magic != b'glTF' or version != 2 or length != path.stat().st_size or kind != 0x4E4F534A or size > length - 20:
                    raise ValueError('Invalid GLB header')
                metadata = json.loads(stream.read(size))
                if not metadata.get('meshes'):
                    raise ValueError('No mesh')
        except (ValueError, OSError, struct.error) as exc:
            invalid.append({'path': str(path.relative_to(RESULTS)), 'error': str(exc)})
    logged = {key(r) for r in generation}
    expected = {key(r) for r in summary}
    report = {
        'scope': 'Imported artifacts and recorded metrics; original generation provenance unverified',
        'models': len(models), 'patent_folders': len({f.parent.name for f in models}),
        'evaluation_rows': len(summary), 'view_rows': len(views), 'generation_rows': len(generation),
        'evaluation_statuses': dict(Counter(r['status'] for r in summary)),
        'generation_statuses': dict(Counter(r['status'] for r in generation)),
        'invalid_glb_headers_or_json': invalid,
        'missing_models': [r['filename'] for r in summary if not (RESULTS / r['patent_id'] / (Path(r['filename']).stem + '.glb')).is_file()],
        'missing_generation_records': [list(k) for k in sorted(expected - logged)],
        'duplicate_evaluation_keys': len(summary) - len(expected),
        'duplicate_generation_keys': len(generation) - len(logged),
        'watertight_counts': dict(Counter(r['watertight'] for r in summary)),
        'zero_best_silhouette_iou': sum(float(r['best_silhouette_iou']) == 0 for r in summary),
        'mean_recorded_generation_seconds': mean(generation, 'generation_seconds'),
        'mean_recorded_peak_vram_gb': mean(generation, 'peak_vram_gb'),
        'mean_best_clip_image_similarity': mean(summary, 'best_clip_image_similarity'),
        'mean_best_clip_caption_similarity': mean(summary, 'best_clip_caption_similarity'),
        'mean_best_ssim': mean(summary, 'best_ssim'),
        'mean_best_silhouette_iou': mean(summary, 'best_silhouette_iou'),
        'mean_connected_components': mean(summary, 'connected_components'),
        'missing_provenance': ['original input sketches/images', 'prompts', 'generation and evaluation scripts', 'model revision and settings', 'GPU model and billing rate'],
        'interpretation': ['Success status means processing completed; it does not establish reconstruction quality.',
                           'CLIP caption scoring does not establish text conditioning during generation.',
                           'Silhouette metrics require alignment and foreground-mask validation before quality conclusions.',
                           'Recorded generation timing scope and memory measurement method are unknown.'],
    }
    output = ROOT / 'experiments/sketch_to_3d/imported_results_audit.json'
    output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
