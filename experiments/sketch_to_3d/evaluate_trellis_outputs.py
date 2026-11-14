#!/usr/bin/env python3
"""
evaluate_trellis_outputs.py

Evaluate IMPACT patent TIFF -> TRELLIS 3D outputs using:
  - SSIM (higher is better)
  - PSNR (higher is better)
  - LPIPS (lower is better)
  - Silhouette IoU (higher is better)
  - Normalized contour Chamfer distance (lower is better)
  - CLIP image-to-image cosine similarity (higher is better)
  - CLIP caption-to-render cosine similarity (higher is better)
  - Multi-view best/mean summaries
  - Mesh vertices/faces, connected components, watertightness,
    non-manifold/boundary edges, degenerate faces, area, volume, extents
  - Per-model evaluation time

This script assumes:
  IMPACT_DIR=/work/11649/schittala/vista/IMPACT
  OUTPUT_DIR=/work/11649/schittala/vista/impact_3d_outputs

It renders each GLB from multiple viewpoints, compares the renders against the
matching TIFF, and writes per-view + per-model CSV files.

For headless Linux rendering, PYOPENGL_PLATFORM is set to "egl" BEFORE
importing pyrender.
"""

import os
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import argparse
import ast
import csv
import math
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.spatial import cKDTree
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
import trimesh
import pyrender
import lpips
from transformers import CLIPModel, CLIPProcessor


DEFAULT_IMPACT_DIR = "/work/11649/schittala/vista/IMPACT"
DEFAULT_OUTPUT_DIR = "/work/11649/schittala/vista/impact_3d_outputs"

# Eight standardized horizontal views. TRELLIS orientation can vary, so
# best-view metrics are useful in addition to mean multi-view metrics.
DEFAULT_VIEWS = [
    ("az000",   0.0, 10.0),
    ("az045",  45.0, 10.0),
    ("az090",  90.0, 10.0),
    ("az135", 135.0, 10.0),
    ("az180", 180.0, 10.0),
    ("az225", 225.0, 10.0),
    ("az270", 270.0, 10.0),
    ("az315", 315.0, 10.0),
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--impact-dir", default=DEFAULT_IMPACT_DIR)
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--patent-id", default=None,
                   help="Evaluate only one patent, e.g. D0908314")
    p.add_argument("--render-size", type=int, default=512)
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--clip-model", default="openai/clip-vit-base-patch32")
    p.add_argument("--views", type=int, default=8, choices=[4, 8],
                   help="Number of horizontal views")
    return p.parse_args()


def resolve_device(name):
    if name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return name


def find_tiff(sample_dir, filename):
    # Fast path if filename is directly under sample_dir.
    direct = Path(sample_dir) / filename
    if direct.exists():
        return str(direct)

    for root, _, files in os.walk(sample_dir):
        if filename in files:
            return str(Path(root) / filename)
    return None


def safe_literal_file_list(value):
    x = ast.literal_eval(value)
    if isinstance(x, str):
        return [x]
    if not isinstance(x, (list, tuple)):
        raise ValueError("file_names is not a list/tuple")
    return [str(v) for v in x]


def load_patent_rows(csv_path):
    rows = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            pid = row.get("id", "").strip()
            if not pid:
                continue
            rows[pid] = row
    return rows


# ---------------------------------------------------------------------------
# Image preprocessing / masks
# ---------------------------------------------------------------------------

def pil_rgb(image_or_path):
    if isinstance(image_or_path, Image.Image):
        return image_or_path.convert("RGB")
    return Image.open(image_or_path).convert("RGB")


def foreground_mask(image):
    """
    Obtain a rough object foreground mask.

    Patent figures and our renders normally have bright backgrounds. We first
    threshold grayscale using Otsu, then choose the polarity that makes border
    pixels mostly background. Small noise is cleaned morphologically.
    """
    rgb = np.asarray(pil_rgb(image))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    _, thresh = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    border = np.concatenate([
        thresh[0, :], thresh[-1, :], thresh[:, 0], thresh[:, -1]
    ])
    border_is_white = np.mean(border > 127) >= 0.5

    # mask True = object
    mask = thresh < 128 if border_is_white else thresh > 127
    mask = (mask.astype(np.uint8) * 255)

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Keep all meaningful components, but drop tiny specks.
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    cleaned = np.zeros_like(mask)
    min_area = max(16, int(0.0002 * mask.shape[0] * mask.shape[1]))
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels == i] = 255

    return cleaned > 0


def bbox_from_mask(mask):
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def crop_and_center(image, size=512, padding_frac=0.08):
    """
    Crop around foreground and resize onto a white square canvas.

    This reduces penalties caused only by different margins/scale and focuses
    the metrics on reconstructed object appearance/shape.
    """
    img = np.asarray(pil_rgb(image))
    mask = foreground_mask(Image.fromarray(img))
    bbox = bbox_from_mask(mask)

    if bbox is None:
        resized = Image.fromarray(img).resize((size, size), Image.LANCZOS)
        return resized, np.zeros((size, size), dtype=bool)

    x1, y1, x2, y2 = bbox
    h, w = img.shape[:2]
    pad_x = int((x2 - x1) * padding_frac)
    pad_y = int((y2 - y1) * padding_frac)
    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(w, x2 + pad_x)
    y2 = min(h, y2 + pad_y)

    crop = img[y1:y2, x1:x2]
    crop_mask = mask[y1:y2, x1:x2]

    ch, cw = crop.shape[:2]
    scale = min(size / max(cw, 1), size / max(ch, 1))
    nw = max(1, int(round(cw * scale)))
    nh = max(1, int(round(ch * scale)))

    crop_img = Image.fromarray(crop).resize((nw, nh), Image.LANCZOS)
    mask_img = Image.fromarray((crop_mask * 255).astype(np.uint8)).resize(
        (nw, nh), Image.NEAREST
    )

    canvas = np.full((size, size, 3), 255, dtype=np.uint8)
    canvas_mask = np.zeros((size, size), dtype=np.uint8)

    x0 = (size - nw) // 2
    y0 = (size - nh) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = np.asarray(crop_img)
    canvas_mask[y0:y0 + nh, x0:x0 + nw] = np.asarray(mask_img)

    return Image.fromarray(canvas), canvas_mask > 0


# ---------------------------------------------------------------------------
# 2D metrics
# ---------------------------------------------------------------------------

def ssim_score(img_a, img_b):
    a = np.asarray(img_a.convert("RGB"))
    b = np.asarray(img_b.convert("RGB"))
    return float(structural_similarity(a, b, channel_axis=2, data_range=255))


def psnr_score(img_a, img_b):
    a = np.asarray(img_a.convert("RGB"))
    b = np.asarray(img_b.convert("RGB"))
    return float(peak_signal_noise_ratio(a, b, data_range=255))


def silhouette_iou(mask_a, mask_b):
    inter = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    if union == 0:
        return float("nan")
    return float(inter / union)


def contour_points(mask):
    contours, _ = cv2.findContours(
        (mask.astype(np.uint8) * 255),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE
    )
    if not contours:
        return np.empty((0, 2), dtype=np.float32)
    pts = np.concatenate([c[:, 0, :] for c in contours], axis=0)
    return pts.astype(np.float32)


def normalized_contour_chamfer(mask_a, mask_b):
    """
    Symmetric contour Chamfer distance normalized by image diagonal.
    Lower is better; 0 means identical contours.
    """
    a = contour_points(mask_a)
    b = contour_points(mask_b)
    if len(a) == 0 or len(b) == 0:
        return float("nan")

    tree_a = cKDTree(a)
    tree_b = cKDTree(b)

    d_a_to_b = tree_b.query(a, k=1)[0].mean()
    d_b_to_a = tree_a.query(b, k=1)[0].mean()
    chamfer = 0.5 * (d_a_to_b + d_b_to_a)

    h, w = mask_a.shape
    diagonal = math.sqrt(w * w + h * h)
    return float(chamfer / diagonal)


def lpips_tensor(img):
    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return t * 2.0 - 1.0


@torch.no_grad()
def lpips_score(model, img_a, img_b, device):
    a = lpips_tensor(img_a).to(device)
    b = lpips_tensor(img_b).to(device)
    return float(model(a, b).item())


# ---------------------------------------------------------------------------
# CLIP metrics
# ---------------------------------------------------------------------------

@torch.no_grad()
def clip_image_embedding(model, processor, image, device):
    inputs = processor(images=image, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)

    output = model.get_image_features(
        pixel_values=pixel_values
    )

    # Newer transformers versions may return BaseModelOutputWithPooling
    if hasattr(output, "pooler_output"):
        feat = output.pooler_output
    else:
        feat = output

    feat = F.normalize(feat, dim=-1)
    return feat


@torch.no_grad()
def clip_text_embedding(model, processor, text, device):
    inputs = processor(
        text=[text],
        padding=True,
        truncation=True,
        return_tensors="pt"
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    output = model.get_text_features(
        input_ids=inputs["input_ids"],
        attention_mask=inputs.get("attention_mask")
    )

    # Newer transformers versions may return BaseModelOutputWithPooling
    if hasattr(output, "pooler_output"):
        feat = output.pooler_output
    else:
        feat = output

    feat = F.normalize(feat, dim=-1)
    return feat

def cosine_from_embeddings(a, b):
    return float((a * b).sum(dim=-1).item())


# ---------------------------------------------------------------------------
# GLB / mesh analysis
# ---------------------------------------------------------------------------

def load_scene_and_mesh(glb_path):
    loaded = trimesh.load(glb_path, force="scene", process=False)

    if isinstance(loaded, trimesh.Trimesh):
        scene = trimesh.Scene(loaded)
    elif isinstance(loaded, trimesh.Scene):
        scene = loaded
    else:
        raise ValueError(f"Unsupported trimesh object: {type(loaded)}")

    # dump(concatenate=True) applies scene transforms and combines geometry.
    mesh = scene.dump(concatenate=True)
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError("Could not concatenate GLB into a triangle mesh")

    return scene, mesh


def edge_manifold_stats(mesh):
    """
    Count unique edges by number of incident faces:
      count == 1 : boundary edge
      count == 2 : manifold interior edge
      count > 2  : non-manifold edge
    """
    if len(mesh.faces) == 0:
        return 0, 0

    inverse = mesh.edges_unique_inverse
    counts = np.bincount(inverse, minlength=len(mesh.edges_unique))
    boundary = int(np.sum(counts == 1))
    nonmanifold = int(np.sum(counts > 2))
    return boundary, nonmanifold


def mesh_metrics(mesh):
    if len(mesh.faces) > 0:
        tri = mesh.triangles
        cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        areas = 0.5 * np.linalg.norm(cross, axis=1)
        degenerate = int(np.sum(areas < 1e-12))
    else:
        degenerate = 0

    boundary_edges, nonmanifold_edges = edge_manifold_stats(mesh)

    try:
        components = mesh.split(only_watertight=False)
        component_count = len(components)
    except Exception:
        component_count = -1

    ext = mesh.extents if len(mesh.vertices) else np.array([np.nan] * 3)
    volume = float(abs(mesh.volume)) if mesh.is_watertight else float("nan")

    return {
        "vertices": int(len(mesh.vertices)),
        "faces": int(len(mesh.faces)),
        "connected_components": int(component_count),
        "watertight": bool(mesh.is_watertight),
        "boundary_edges": boundary_edges,
        "nonmanifold_edges": nonmanifold_edges,
        "degenerate_faces": degenerate,
        "surface_area": float(mesh.area),
        "volume_if_watertight": volume,
        "bbox_x": float(ext[0]),
        "bbox_y": float(ext[1]),
        "bbox_z": float(ext[2]),
    }


# ---------------------------------------------------------------------------
# Multi-view rendering
# ---------------------------------------------------------------------------

def camera_pose(position, target):
    position = np.asarray(position, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)

    forward = target - position
    forward /= np.linalg.norm(forward)

    up_guess = np.array([0.0, 1.0, 0.0])
    if abs(np.dot(forward, up_guess)) > 0.98:
        up_guess = np.array([0.0, 0.0, 1.0])

    right = np.cross(forward, up_guess)
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    up /= np.linalg.norm(up)

    pose = np.eye(4)
    pose[:3, 0] = right
    pose[:3, 1] = up
    pose[:3, 2] = -forward  # OpenGL camera looks down -Z
    pose[:3, 3] = position
    return pose


def normalize_scene_for_render(scene):
    """
    Copy the scene and translate/scale it into a predictable centered volume.
    """
    s = scene.copy()
    bounds = s.bounds
    if bounds is None:
        raise ValueError("Scene has no bounds")

    center = bounds.mean(axis=0)
    extents = bounds[1] - bounds[0]
    max_extent = float(np.max(extents))
    if max_extent <= 0:
        raise ValueError("Degenerate scene bounds")

    # Apply a root transform: center then normalize longest dimension to 2.
    T = np.eye(4)
    T[:3, 3] = -center

    S = np.eye(4)
    scale = 2.0 / max_extent
    S[:3, :3] *= scale

    # Reconstruct transformed scene geometry because assigning a root transform
    # is not consistently supported across trimesh versions.
    out = trimesh.Scene()
    for name, geom in s.geometry.items():
        # geometry nodes can have transforms; easiest reliable path is dump().
        pass

    combined = s.dump(concatenate=True)
    combined.apply_translation(-center)
    combined.apply_scale(scale)
    return trimesh.Scene(combined)


def render_view(scene, azimuth_deg, elevation_deg, size):
    normalized = normalize_scene_for_render(scene)

    pr_scene = pyrender.Scene.from_trimesh_scene(
        normalized,
        bg_color=np.array([255, 255, 255, 255], dtype=np.uint8),
        ambient_light=np.array([0.55, 0.55, 0.55])
    )

    radius = 3.2
    az = math.radians(azimuth_deg)
    el = math.radians(elevation_deg)

    # Y is treated as vertical.
    position = np.array([
        radius * math.sin(az) * math.cos(el),
        radius * math.sin(el),
        radius * math.cos(az) * math.cos(el),
    ])

    pose = camera_pose(position, np.zeros(3))

    camera = pyrender.PerspectiveCamera(
        yfov=math.radians(40.0),
        aspectRatio=1.0
    )
    pr_scene.add(camera, pose=pose)

    # Add three directional lights for readable geometry without harsh shadows.
    for light_pose in (
        pose,
        camera_pose([3.0, 3.0, 3.0], [0, 0, 0]),
        camera_pose([-3.0, 2.0, 2.0], [0, 0, 0]),
    ):
        light = pyrender.DirectionalLight(color=np.ones(3), intensity=2.0)
        pr_scene.add(light, pose=light_pose)

    renderer = pyrender.OffscreenRenderer(size, size)
    try:
        color, _ = renderer.render(
            pr_scene,
            flags=pyrender.RenderFlags.RGBA
        )
    finally:
        renderer.delete()

    rgb = color[:, :, :3]
    return Image.fromarray(rgb.astype(np.uint8), mode="RGB")


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_one(
    patent_id,
    filename,
    tiff_path,
    glb_path,
    caption,
    render_dir,
    render_size,
    views,
    device,
    lpips_model,
    clip_model,
    clip_processor,
):
    start = time.perf_counter()

    original = pil_rgb(tiff_path)
    original_aligned, original_mask = crop_and_center(original, render_size)

    # Compute invariant original embedding once.
    original_clip = clip_image_embedding(
        clip_model, clip_processor, original_aligned, device
    )
    caption_clip = clip_text_embedding(
        clip_model, clip_processor, caption, device
    ) if caption else None

    scene, mesh = load_scene_and_mesh(glb_path)
    mmetrics = mesh_metrics(mesh)

    Path(render_dir).mkdir(parents=True, exist_ok=True)
    original_aligned.save(Path(render_dir) / "reference_aligned.png")

    per_view = []

    for view_name, azimuth, elevation in views:
        render = render_view(scene, azimuth, elevation, render_size)
        render_path = Path(render_dir) / f"{view_name}.png"
        render.save(render_path)

        render_aligned, render_mask = crop_and_center(render, render_size)
        render_aligned.save(Path(render_dir) / f"{view_name}_aligned.png")

        rclip = clip_image_embedding(
            clip_model, clip_processor, render_aligned, device
        )

        row = {
            "patent_id": patent_id,
            "filename": filename,
            "glb": str(glb_path),
            "view": view_name,
            "azimuth_deg": azimuth,
            "elevation_deg": elevation,
            "ssim": ssim_score(original_aligned, render_aligned),
            "psnr": psnr_score(original_aligned, render_aligned),
            "lpips": lpips_score(
                lpips_model, original_aligned, render_aligned, device
            ),
            "silhouette_iou": silhouette_iou(
                original_mask, render_mask
            ),
            "contour_chamfer_norm": normalized_contour_chamfer(
                original_mask, render_mask
            ),
            "clip_image_similarity": cosine_from_embeddings(
                original_clip, rclip
            ),
            "clip_caption_similarity": (
                cosine_from_embeddings(caption_clip, rclip)
                if caption_clip is not None else float("nan")
            ),
            "render_path": str(render_path),
        }
        per_view.append(row)

    elapsed = time.perf_counter() - start

    # Summary: mean over all views plus best view for each criterion.
    def vals(key):
        return np.asarray([r[key] for r in per_view], dtype=np.float64)

    summary = {
        "patent_id": patent_id,
        "filename": filename,
        "tiff_path": str(tiff_path),
        "glb_path": str(glb_path),
        **mmetrics,
        "mean_ssim": float(np.nanmean(vals("ssim"))),
        "best_ssim": float(np.nanmax(vals("ssim"))),
        "mean_psnr": float(np.nanmean(vals("psnr"))),
        "best_psnr": float(np.nanmax(vals("psnr"))),
        "mean_lpips": float(np.nanmean(vals("lpips"))),
        "best_lpips": float(np.nanmin(vals("lpips"))),
        "mean_silhouette_iou": float(np.nanmean(vals("silhouette_iou"))),
        "best_silhouette_iou": float(np.nanmax(vals("silhouette_iou"))),
        "mean_contour_chamfer_norm": float(
            np.nanmean(vals("contour_chamfer_norm"))
        ),
        "best_contour_chamfer_norm": float(
            np.nanmin(vals("contour_chamfer_norm"))
        ),
        "mean_clip_image_similarity": float(
            np.nanmean(vals("clip_image_similarity"))
        ),
        "best_clip_image_similarity": float(
            np.nanmax(vals("clip_image_similarity"))
        ),
        "mean_clip_caption_similarity": float(
            np.nanmean(vals("clip_caption_similarity"))
        ),
        "best_clip_caption_similarity": float(
            np.nanmax(vals("clip_caption_similarity"))
        ),
        "best_ssim_view": per_view[int(np.nanargmax(vals("ssim")))]["view"],
        "best_clip_view": per_view[
            int(np.nanargmax(vals("clip_image_similarity")))
        ]["view"],
        "evaluation_seconds": float(elapsed),
        "status": "success",
        "error": "",
    }

    return per_view, summary


def write_csv(path, rows):
    if not rows:
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    keys = []
    seen = set()
    for row in rows:
        for k in row:
            if k not in seen:
                seen.add(k)
                keys.append(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    device = resolve_device(args.device)
    print(f"Device: {device}")

    impact_dir = Path(args.impact_dir)
    sample_dir = impact_dir / "Sample data"
    csv_path = sample_dir / "sample_data.csv"
    output_dir = Path(args.output_dir)

    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    print("Loading LPIPS...")
    lpips_model = lpips.LPIPS(net="alex").to(device).eval()

    print(f"Loading CLIP: {args.clip_model}")
    clip_model = CLIPModel.from_pretrained(args.clip_model, use_safetensors=False).to(device).eval()
    clip_processor = CLIPProcessor.from_pretrained(args.clip_model)

    rows = load_patent_rows(csv_path)

    if args.patent_id:
        if args.patent_id not in rows:
            raise KeyError(f"{args.patent_id} not found in CSV")
        selected = [(args.patent_id, rows[args.patent_id])]
    else:
        selected = list(rows.items())

    views = DEFAULT_VIEWS if args.views == 8 else DEFAULT_VIEWS[::2]

    all_view_rows = []
    all_summary_rows = []

    attempted = 0
    successful = 0

    for patent_id, row in selected:
        caption = row.get("caption", "").strip()

        try:
            filenames = safe_literal_file_list(row.get("file_names", "[]"))
        except Exception as e:
            print(f"[SKIP] {patent_id}: bad file_names: {e}")
            continue

        patent_output = output_dir / patent_id

        for filename in filenames:
            image_id = Path(filename).stem
            tiff_path = find_tiff(sample_dir, filename)
            glb_path = patent_output / f"{image_id}.glb"

            attempted += 1

            if tiff_path is None:
                print(f"[MISSING TIFF] {filename}")
                all_summary_rows.append({
                    "patent_id": patent_id,
                    "filename": filename,
                    "status": "failed",
                    "error": "TIFF not found",
                })
                continue

            if not glb_path.exists():
                print(f"[MISSING GLB] {glb_path}")
                all_summary_rows.append({
                    "patent_id": patent_id,
                    "filename": filename,
                    "tiff_path": tiff_path,
                    "glb_path": str(glb_path),
                    "status": "failed",
                    "error": "GLB not found",
                })
                continue

            print()
            print("=" * 80)
            print(f"Evaluating: {patent_id} / {filename}")
            print("=" * 80)

            render_dir = patent_output / "evaluation" / image_id

            try:
                per_view, summary = evaluate_one(
                    patent_id=patent_id,
                    filename=filename,
                    tiff_path=tiff_path,
                    glb_path=glb_path,
                    caption=caption,
                    render_dir=render_dir,
                    render_size=args.render_size,
                    views=views,
                    device=device,
                    lpips_model=lpips_model,
                    clip_model=clip_model,
                    clip_processor=clip_processor,
                )

                all_view_rows.extend(per_view)
                all_summary_rows.append(summary)
                successful += 1

                print(
                    f"  best SSIM={summary['best_ssim']:.4f} | "
                    f"best IoU={summary['best_silhouette_iou']:.4f} | "
                    f"best LPIPS={summary['best_lpips']:.4f} | "
                    f"best CLIP={summary['best_clip_image_similarity']:.4f}"
                )

            except Exception as e:
                print(f"[ERROR] {filename}: {type(e).__name__}: {e}")
                all_summary_rows.append({
                    "patent_id": patent_id,
                    "filename": filename,
                    "tiff_path": tiff_path,
                    "glb_path": str(glb_path),
                    "status": "failed",
                    "error": f"{type(e).__name__}: {e}",
                })

            # Continuously save progress so a long HPC job can be interrupted.
            write_csv(output_dir / "metrics_per_view.csv", all_view_rows)
            write_csv(output_dir / "metrics_summary.csv", all_summary_rows)

    success_rate = successful / attempted if attempted else float("nan")

    pipeline_summary = [{
        "attempted_tiffs": attempted,
        "successful_evaluations": successful,
        "failed_evaluations": attempted - successful,
        "evaluation_success_rate": success_rate,
    }]
    write_csv(output_dir / "evaluation_pipeline_summary.csv", pipeline_summary)

    print()
    print("Finished.")
    print(f"Attempted:  {attempted}")
    print(f"Successful: {successful}")
    print(f"Success rate: {100.0 * success_rate:.2f}%")
    print(f"Summary CSV: {output_dir / 'metrics_summary.csv'}")
    print(f"Per-view CSV: {output_dir / 'metrics_per_view.csv'}")


if __name__ == "__main__":
    main()
