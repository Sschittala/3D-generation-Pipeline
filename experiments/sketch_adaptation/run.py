"""Matched sketch-input ablations. Preparation is CPU-only; generation requires TRELLIS CUDA."""
import argparse
import csv
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

VARIANTS = ('stock', 'bypass', 'distance', 'distance_blur')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def read_manifest(path):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError('Manifest is empty')
    seen, patents = set(), {}
    for r in rows:
        for key in ('sample_id', 'patent_id'):
            if not r.get(key) or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in r[key]):
                raise ValueError(f'{key} must use letters, digits, underscore or hyphen')
        if r['sample_id'] in seen:
            raise ValueError('Duplicate sample_id')
        seen.add(r['sample_id'])
        if r.get('split') not in ('dev', 'test'):
            raise ValueError('Every sample requires split=dev or test')
        if patents.setdefault(r['patent_id'], r['split']) != r['split']:
            raise ValueError('A patent cannot appear in both dev and test')
        source = Path(r['image']).expanduser()
        if not source.is_absolute():
            source = path.parent / source
        r['image'] = str(source.resolve())
        if not source.is_file():
            raise ValueError(f'Missing image: {source}')
        if r.get('crop_reviewed') is not True:
            raise ValueError('Inspect each crop and set crop_reviewed=true before preparing inputs')
        box = r.get('crop')
        if not isinstance(box, list) or len(box) != 4 or any(type(v) is not int for v in box):
            raise ValueError('Every sample requires an explicit integer [left,top,right,bottom] crop')
    return rows


def prepare_images(image, size, width, blur):
    """Common fixed canvas; soft distance-to-ink map, not a probability or silhouette."""
    import cv2
    # Respect shared-node thread limits; OMP settings do not limit every OpenCV backend.
    cv2.setNumThreads(1)
    import numpy as np
    from PIL import Image, ImageOps
    image = ImageOps.contain(image.convert('RGB'), (size, size), Image.Resampling.LANCZOS)
    canvas = Image.new('RGB', (size, size), 'white')
    canvas.paste(image, ((size-image.width)//2, (size-image.height)//2))
    gray = np.asarray(canvas.convert('L'))
    threshold, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    ink = gray <= threshold
    if not ink.any() or ink.all():
        raise ValueError('Empty or uniform ink mask; inspect source polarity/crop')
    # No opening, erosion, or small-component removal: preserve thin strokes.
    distance = cv2.distanceTransform((~ink).astype('uint8'), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    soft = np.exp(-.5*(distance/width)**2)
    smooth = cv2.GaussianBlur(soft, (0, 0), blur)
    convert = lambda x: Image.fromarray(np.clip(np.rint(255*(1-x)), 0, 255).astype('uint8')).convert('RGB')
    return {'reference':canvas, 'ink_mask':Image.fromarray((ink*255).astype('uint8')),
            'distance':convert(soft), 'distance_blur':convert(smooth)}, float(threshold)


def prepare(args):
    from PIL import Image
    rows = read_manifest(args.manifest.resolve())
    if args.output.exists():
        raise ValueError('Prepared directory exists; use a new directory to preserve the protocol')
    if args.size < 32 or args.width <= 0 or args.blur <= 0:
        raise ValueError('Size >=32 and positive map width / blur are required')
    # Validate crops and patent captions before creating output.
    captions={}
    for r in rows:
        if captions.setdefault(r['patent_id'],r.get('caption','')) != r.get('caption',''):
            raise ValueError('The evaluator requires one shared caption per patent')
        with Image.open(r['image']) as image:
            l,t,right,bottom = r['crop']
            if not (0 <= l < right <= image.width and 0 <= t < bottom <= image.height):
                raise ValueError(f'Invalid crop for {r["sample_id"]}')
    args.output.mkdir(parents=True)
    write_json(args.output/'protocol.json', dict(size=args.size, distance_width_pixels=args.width,
        blur_sigma_pixels=args.blur, manifest_sha256=digest(args.manifest), variants=VARIANTS,
        note='Parameters must be fixed on development patents before test evaluation.'))
    for r in rows:
        sample_dir=args.output/'inputs'/r['sample_id'];sample_dir.mkdir(parents=True)
        with Image.open(r['image']) as image:
            crop=image.convert('RGB').crop(r['crop']);crop.save(sample_dir/'crop.png')
            images,threshold=prepare_images(crop,args.size,args.width,args.blur)
        for name,image in images.items():
            image.save(sample_dir/(name+'.png'))
        r['source_sha256']=digest(Path(r['image']))
        r['otsu_threshold']=threshold
        r['prepared_hashes']={name:digest(sample_dir/(name+'.png')) for name in images}
    (args.output/'manifest.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    # Original evaluator reads one metadata row per patent; create a common,
    # untransformed reference dataset for each split. No adapted references.
    for split in ('dev','test'):
        destination=args.output/split/'IMPACT'/'Sample data';destination.mkdir(parents=True)
        patents={}
        for r in rows:
            if r['split'] != split: continue
            (destination/(r['sample_id']+'.png')).write_bytes((args.output/'inputs'/r['sample_id']/'reference.png').read_bytes())
            entry=patents.setdefault(r['patent_id'], {'id':r['patent_id'],'caption':r.get('caption',''),'file_names':[]})
            if entry['caption'] != r.get('caption',''):
                raise ValueError('The existing evaluator requires one shared caption per patent')
            entry['file_names'].append(r['sample_id']+'.png')
        with (destination/'sample_data.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=['id','caption','file_names']);writer.writeheader()
            for entry in patents.values():
                entry['file_names']=repr(entry['file_names']);writer.writerow(entry)
    print(f'Prepared {len(rows)} samples. Inspect every crop/map before GPU generation: {args.output}')


def generate(args):
    rows=[json.loads(line) for line in (args.prepared/'manifest.jsonl').read_text().splitlines()]
    rows=[r for r in rows if r['split']==args.split]
    if not rows: raise ValueError('No samples in requested split')
    if len(set(args.seeds)) != len(args.seeds) or len(set(args.variants)) != len(args.variants):
        raise ValueError('Seeds and variants must be unique')
    if not (args.trellis_root/'trellis/pipelines/trellis_image_to_3d.py').is_file():
        raise ValueError('TRELLIS source root is missing trellis/pipelines/trellis_image_to_3d.py')
    plan=dict(prepared=str(args.prepared.resolve()),protocol_sha256=digest(args.prepared/'protocol.json'),
              manifest_sha256=digest(args.prepared/'manifest.jsonl'),split=args.split,model=args.model,
              variants=args.variants,seeds=args.seeds,samples=len(rows),trellis_root=str(args.trellis_root.resolve()))
    plan['attention_backend']=getattr(args,'attention_backend','installed')
    if args.dry_run:
        print(json.dumps(plan,indent=2));return
    if args.output.exists(): raise ValueError('Run directory exists; choose a new directory')
    sys.path.insert(0,str(args.trellis_root.resolve()))
    import torch
    from PIL import Image
    if plan['attention_backend']=='sdpa':
        if __package__:
            from .sdpa_compat import install
        else:
            from sdpa_compat import install
        install()
        plan['sdpa_adapter_sha256']=digest(Path(__file__).with_name('sdpa_compat.py'))
    from trellis.pipelines import TrellisImageTo3DPipeline
    from trellis.utils import postprocessing_utils
    if not torch.cuda.is_available(): raise ValueError('CUDA is required')
    args.output.mkdir(parents=True)
    plan.update(gpu=torch.cuda.get_device_name(0),platform=platform.platform(),versions={},runner_sha256=digest(Path(__file__)))
    for package in ('torch','numpy','Pillow','opencv-python','transformers','rembg'):
        try: plan['versions'][package]=importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError: pass
    try: plan['trellis_commit']=subprocess.check_output(['git','-C',str(args.trellis_root),'rev-parse','HEAD'],text=True).strip()
    except subprocess.CalledProcessError: plan['trellis_commit']='unavailable'
    source=args.trellis_root/'trellis/pipelines/trellis_image_to_3d.py'
    if source.exists(): plan['pipeline_source_sha256']=digest(source)
    plan['model_setup_status']='loading'
    write_json(args.output/'run.json',plan)
    setup=time.perf_counter()
    try:
        pipeline=TrellisImageTo3DPipeline.from_pretrained(args.model);pipeline.cuda()
    except Exception as exc:
        plan.update(model_setup_status='failed',model_setup_error=f'{type(exc).__name__}: {exc}')
        write_json(args.output/'run.json',plan)
        raise
    torch.cuda.synchronize();plan['model_setup_seconds']=time.perf_counter()-setup
    plan['model_setup_status']='success'
    plan['sparse_structure_sampler_params']=pipeline.sparse_structure_sampler_params
    plan['slat_sampler_params']=pipeline.slat_sampler_params
    write_json(args.output/'run.json',plan)
    failures=0
    for variant in args.variants:
        for seed in args.seeds:
            for row in rows:
                sample=args.prepared/'inputs'/row['sample_id']
                name={'stock':'reference','bypass':'reference','distance':'distance','distance_blur':'distance_blur'}[variant]
                source=sample/(name+'.png')
                if digest(source) != row['prepared_hashes'][name]:
                    raise ValueError('Prepared input changed after preparation')
                destination=args.output/variant/f'seed-{seed}'/row['patent_id'];destination.mkdir(parents=True,exist_ok=True)
                record=dict(sample_id=row['sample_id'],patent_id=row['patent_id'],variant=variant,seed=seed,
                            input_sha256=digest(source),preprocess_image=variant=='stock')
                start=time.perf_counter()
                try:
                    image=Image.open(source).convert('RGB')
                    torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize()
                    stage=time.perf_counter()
                    with torch.inference_mode():
                        result=pipeline.run(image,seed=seed,preprocess_image=variant=='stock',formats=['mesh','gaussian'])
                    torch.cuda.synchronize();record['pipeline_seconds']=time.perf_counter()-stage
                    record['pipeline_peak_allocated_gib']=torch.cuda.max_memory_allocated()/2**30
                    stage=time.perf_counter()
                    glb=postprocessing_utils.to_glb(result['gaussian'][0],result['mesh'][0],simplify=.95,texture_size=1024)
                    glb.export(str(destination/(row['sample_id']+'.glb')))
                    torch.cuda.synchronize();record['export_seconds']=time.perf_counter()-stage
                    record['status']='success'
                    del result,glb
                except Exception as exc:
                    failures+=1;record.update(status='failed',error=f'{type(exc).__name__}: {exc}')
                record['total_sample_seconds']=time.perf_counter()-start
                write_json(destination/(row['sample_id']+'.run.json'),record)
                print(f'{variant} seed={seed} {row["sample_id"]}: {record["status"]}',flush=True)
    if failures: raise SystemExit(f'{failures} generations failed; inspect .run.json logs. Do not drop failures from reporting.')


def main():
    parser=argparse.ArgumentParser(description=__doc__);commands=parser.add_subparsers(dest='command',required=True)
    p=commands.add_parser('prepare');p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--size',type=int,default=518);p.add_argument('--width',type=float,default=2.0);p.add_argument('--blur',type=float,default=1.0)
    p=commands.add_parser('generate');p.add_argument('--prepared',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--trellis-root',type=Path,required=True);p.add_argument('--model',default='microsoft/TRELLIS-image-large')
    p.add_argument('--split',choices=['dev','test'],required=True);p.add_argument('--seeds',nargs='+',type=int,default=[1,2,3])
    p.add_argument('--variants',nargs='+',choices=VARIANTS,default=list(VARIANTS));p.add_argument('--dry-run',action='store_true')
    p.add_argument('--attention-backend',choices=['installed','sdpa'],default='installed',
                   help='sdpa uses the opt-in TRELLIS adapter without external flash-attn/xformers')
    args=parser.parse_args()
    try:
        (prepare if args.command=='prepare' else generate)(args)
    except (ValueError,FileNotFoundError) as exc:
        parser.error(str(exc))


if __name__=='__main__': main()
