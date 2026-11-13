"""CUDA experiment stages. Each invocation loads only one model family."""
import argparse
import hashlib
import importlib.metadata
import json
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['image', 'trellis'])
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='New directory; existing results are never overwritten')
    parser.add_argument('--prompt', help='Required for image stage; unsupported by TRELLIS image-to-3D')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--steps', type=int, default=30)
    parser.add_argument('--control-scale', type=float, default=1.0)
    parser.add_argument('--base-model', default='stable-diffusion-v1-5/stable-diffusion-v1-5')
    parser.add_argument('--control-model', default='lllyasviel/control_v11p_sd15_lineart')
    parser.add_argument('--trellis-model', default='microsoft/TRELLIS-image-large')
    parser.add_argument('--gpu-hourly-rate', type=float, help='Optional user-supplied USD/hour, excludes idle time')
    parser.add_argument('--dry-run', action='store_true', help='Validate inputs and print plan without importing models')
    args = parser.parse_args()
    if not args.input.is_file():
        parser.error(f'Input does not exist: {args.input}')
    if args.output.exists():
        parser.error('Output directory already exists; choose a new directory')
    if args.stage == 'image' and not (args.prompt and args.prompt.strip()):
        parser.error('The image stage requires --prompt')
    if args.stage == 'trellis' and args.prompt is not None:
        parser.error('TRELLIS image-to-3D does not accept text; omit --prompt')
    if args.steps <= 0 or args.control_scale < 0 or (args.gpu_hourly_rate is not None and args.gpu_hourly_rate < 0):
        parser.error('Steps must be positive; control scale and rate must be nonnegative')
    record = {k: str(v.resolve()) if isinstance(v, Path) else v for k, v in vars(args).items()}
    record.update(text_conditioned=args.stage == 'image', input_sha256=hashlib.sha256(args.input.read_bytes()).hexdigest(), host=platform.platform())
    if args.dry_run:
        print(json.dumps(record, indent=2))
        return
    import torch
    from PIL import Image, ImageOps
    if not torch.cuda.is_available():
        parser.error('A CUDA GPU and compatible environment are required; no output was created')
    record['gpu'] = torch.cuda.get_device_name(0)
    record['versions'] = {}
    for package in ['torch', 'diffusers', 'transformers', 'Pillow', 'numpy']:
        try:
            record['versions'][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    args.output.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    try:
        image = Image.open(args.input).convert('RGB')
        if args.stage == 'image':
            from diffusers import ControlNetModel, StableDiffusionControlNetPipeline
            control = ControlNetModel.from_pretrained(args.control_model, torch_dtype=torch.float16)
            pipeline = StableDiffusionControlNetPipeline.from_pretrained(args.base_model, controlnet=control, torch_dtype=torch.float16).to('cuda')
            # Lineart checkpoint expects dark lines on a light background. Preserve aspect ratio.
            image = ImageOps.pad(image, (512, 512), color='white')
            image.save(args.output / 'control_input.png')
        else:
            sys.path.insert(0, str(ROOT / 'TRELLIS'))
            from trellis.pipelines import TrellisImageTo3DPipeline
            from trellis.utils import postprocessing_utils
            pipeline = TrellisImageTo3DPipeline.from_pretrained(args.trellis_model)
            pipeline.cuda()
        torch.cuda.synchronize()
        record['setup_seconds'] = time.perf_counter() - start
        torch.cuda.reset_peak_memory_stats()
        inference_start = time.perf_counter()
        with torch.inference_mode():
            if args.stage == 'image':
                result = pipeline(prompt=args.prompt, image=image, num_inference_steps=args.steps,
                                  controlnet_conditioning_scale=args.control_scale,
                                  generator=torch.Generator(device='cuda').manual_seed(args.seed))
            else:
                result = pipeline.run(image, seed=args.seed, formats=['mesh', 'gaussian'])
        torch.cuda.synchronize()
        record['inference_seconds'] = time.perf_counter() - inference_start
        record['peak_allocated_vram_gib'] = torch.cuda.max_memory_allocated() / 2**30
        record['peak_reserved_vram_gib'] = torch.cuda.max_memory_reserved() / 2**30
        export_start = time.perf_counter()
        if args.stage == 'image':
            if result.nsfw_content_detected and any(result.nsfw_content_detected):
                raise RuntimeError('Image safety checker flagged output; generation was not accepted')
            result.images[0].save(args.output / 'image.png')
        else:
            glb = postprocessing_utils.to_glb(result['gaussian'][0], result['mesh'][0], simplify=0.95, texture_size=1024)
            glb.export(str(args.output / 'model.glb'))
        torch.cuda.synchronize()
        record['export_seconds'] = time.perf_counter() - export_start
        record['status'] = 'success'
    except Exception as exc:
        record.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        record['total_seconds'] = time.perf_counter() - start
        if args.gpu_hourly_rate is not None:
            record['estimated_active_cost_usd'] = record['total_seconds'] / 3600 * args.gpu_hourly_rate
        (args.output / 'run.json').write_text(json.dumps(record, indent=2) + '\n')


if __name__ == '__main__':
    main()
