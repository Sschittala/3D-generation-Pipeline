"""Run the existing evaluator against common references; summarize matched ablations."""
import argparse
import csv
import hashlib
import json
import math
import random
import statistics
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

METRICS = ['mean_ssim','mean_psnr','mean_lpips','mean_clip_image_similarity','mean_clip_caption_similarity']


def load_run(run):
    config=json.loads((run/'run.json').read_text())
    prepared=Path(config['prepared'])
    for name,key in [('protocol.json','protocol_sha256'),('manifest.jsonl','manifest_sha256')]:
        if hashlib.sha256((prepared/name).read_bytes()).hexdigest()!=config[key]:
            raise ValueError('Prepared protocol or manifest changed after generation')
    rows=[json.loads(line) for line in (prepared/'manifest.jsonl').read_text().splitlines()]
    rows=[r for r in rows if r['split']==config['split']]
    return config,prepared,rows


def evaluate(args):
    config,prepared,rows=load_run(args.run)
    impact=prepared/config['split']/'IMPACT'
    for row in rows:
        reference=impact/'Sample data'/(row['sample_id']+'.png')
        if hashlib.sha256(reference.read_bytes()).hexdigest()!=row['prepared_hashes']['reference']:
            raise ValueError('Evaluation reference changed after preparation')
    for variant in config['variants']:
        for seed in config['seeds']:
            output=args.run/variant/f'seed-{seed}'
            command=[sys.executable,str(args.evaluator.resolve()),'--impact-dir',str(impact),
                     '--output-dir',str(output.resolve()),'--views','8','--render-size','512',
                     '--clip-model',args.clip_model,'--device',args.device]
            print(' '.join(command),flush=True)
            if not args.dry_run:
                if (output/'metrics_summary.csv').exists():
                    raise ValueError('Existing evaluation CSV: use a fresh run or inspect it; no automatic overwrite')
                output.mkdir(parents=True,exist_ok=True)
                (output/'evaluation_config.json').write_text(json.dumps(dict(command=command,evaluator_sha256=hashlib.sha256(args.evaluator.read_bytes()).hexdigest(),reference_manifest_sha256=config['manifest_sha256']),indent=2)+'\n')
                subprocess.run(command,check=True)


def bootstrap(deltas, repeats=2000):
    if len(deltas)<2: return None,None
    rng=random.Random(42)
    means=sorted(statistics.mean(rng.choices(deltas,k=len(deltas))) for _ in range(repeats))
    return means[int(.025*(repeats-1))],means[int(.975*(repeats-1))]


def summarize(args):
    config,prepared,rows=load_run(args.run)
    variants=config['variants'];seeds=config['seeds']
    if 'stock' not in variants: raise ValueError('Include stock for paired comparisons')
    measurements={};coverage=[]
    for variant in variants:
        for seed in seeds:
            folder=args.run/variant/f'seed-{seed}'
            path=folder/'metrics_summary.csv'
            table={}
            if path.exists():
                with path.open() as f:
                    for row in csv.DictReader(f):
                        key=(row['patent_id'],Path(row['filename']).stem)
                        if key in table: raise ValueError(f'Duplicate evaluation key: {path} {key}')
                        table[key]=row
            generated=evaluated=0
            for r in rows:
                key=(r['patent_id'],r['sample_id'])
                log=folder/r['patent_id']/(r['sample_id']+'.run.json')
                generation_ok=log.exists() and json.loads(log.read_text()).get('status')=='success'
                if generation_ok: generated+=1
                row=table.get(key,{})
                if row.get('status')=='success':
                    evaluated+=1
                    if generation_ok: measurements[(variant,seed,*key)]=row
            coverage.append(dict(variant=variant,seed=seed,planned=len(rows),generation_success=generated,
                                 evaluation_success=evaluated,not_successfully_evaluated=len(rows)-evaluated))
    summaries=[]
    for metric in METRICS:
        eligible=[]
        for r in rows:
            key=(r['patent_id'],r['sample_id'])
            def valid(variant,seed):
                value=measurements.get((variant,seed,*key),{}).get(metric,'')
                try:return math.isfinite(float(value))
                except (ValueError,TypeError):return False
            if all(valid(variant,seed) for variant in variants for seed in seeds): eligible.append(key)
        grouped={variant:defaultdict(list) for variant in variants}
        for patent,sample in eligible:
            for variant in variants:
                seed_mean=statistics.mean(float(measurements[(variant,seed,patent,sample)][metric]) for seed in seeds)
                grouped[variant][patent].append(seed_mean)
        patent_means={variant:{patent:statistics.mean(vals) for patent,vals in groups.items()} for variant,groups in grouped.items()}
        for variant in variants:
            values=list(patent_means[variant].values())
            deltas=[value-patent_means['stock'][patent] for patent,value in patent_means[variant].items()]
            low,high=bootstrap(deltas) if variant!='stock' else (None,None)
            summaries.append(dict(metric=metric,variant=variant,matched_images=len(eligible),matched_patents=len(values),
                mean=statistics.mean(values) if values else None,sd=statistics.stdev(values) if len(values)>1 else None,
                delta_vs_stock=statistics.mean(deltas) if deltas else None,delta_ci_low=low,delta_ci_high=high))
    output=args.run/'analysis';output.mkdir(exist_ok=True)
    for name,records in [('coverage',coverage),('paired_metrics',summaries)]:
        with (output/(name+'.csv')).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    lines=['# Matched ablation results','','All values are newly evaluated outputs, when available. No scores are simulated or adjusted. Each metric uses only samples with finite scores for every variant and seed. Seeds are averaged within a sample, samples within a patent, and patents weighted equally. Coverage exclusions are reported separately; the complete-case table does not measure end-to-end success.','','| Metric | Variant | Images / patents | Mean ± patent SD | Change vs stock [95% CI] |','| --- | --- | --- | --- | --- |']
    fmt=lambda value: 'NA' if value is None else f'{value:.4f}'
    for r in summaries:
        delta='Reference' if r['variant']=='stock' else f"{fmt(r['delta_vs_stock'])} [{fmt(r['delta_ci_low'])}, {fmt(r['delta_ci_high'])}]"
        lines.append(f"| {r['metric']} | {r['variant']} | {r['matched_images']} / {r['matched_patents']} | {fmt(r['mean'])} ± {fmt(r['sd'])} | {delta} |")
    lines+=['','Intervals resample paired patent-level differences 2,000 times with fixed bootstrap seed 42. They reflect variation across sampled patents, not an independent estimate of seed uncertainty. With fewer than two matched patents intervals are NA. Lower LPIPS is better; for the other listed metrics higher is better. These exploratory intervals are not adjusted for multiple comparisons. Report all ablations rather than selecting only favorable comparisons.','','## Coverage','','| Variant | Seed | Planned | Generated | Evaluated | Not evaluated successfully |','| --- | --- | --- | --- | --- | --- |']
    lines+=['| '+' | '.join(map(str,r.values()))+' |' for r in coverage]
    lines+=['','Silhouette metrics are omitted pending validated masks and camera correspondence. No geometric ground truth is implied. Test results support claims only after confirming input crops, preprocessing parameters, data splits and run provenance.']
    (output/'results.md').write_text('\n'.join(lines)+'\n')
    print(output/'results.md')


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    e=sub.add_parser('evaluate');e.add_argument('--run',type=Path,required=True);e.add_argument('--evaluator',type=Path,required=True)
    e.add_argument('--clip-model',default='openai/clip-vit-base-patch32');e.add_argument('--device',choices=['cuda','cpu','auto'],default='cuda');e.add_argument('--dry-run',action='store_true')
    s=sub.add_parser('summarize');s.add_argument('--run',type=Path,required=True)
    args=p.parse_args()
    try: (evaluate if args.command=='evaluate' else summarize)(args)
    except ValueError as exc: p.error(str(exc))


if __name__=='__main__': main()
