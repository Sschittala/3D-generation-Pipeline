import os
import csv
import ast
import time
import traceback

import torch
from PIL import Image

from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils import postprocessing_utils


# ============================================================
# PATHS
# ============================================================

IMPACT_DIR = "/work/11649/schittala/vista/IMPACT"

SAMPLE_DIR = os.path.join(
    IMPACT_DIR,
    "Sample data"
)

CSV_PATH = os.path.join(
    SAMPLE_DIR,
    "sample_data.csv"
)

OUTPUT_DIR = (
    "/work/11649/schittala/vista/"
    "impact_3d_outputs"
)

GENERATION_CSV = os.path.join(
    OUTPUT_DIR,
    "generation_metrics.csv"
)


# ============================================================
# ENVIRONMENT
# ============================================================

os.environ["SPCONV_ALGO"] = "native"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# FIND TIFF
# ============================================================

def find_tiff(filename):

    direct_path = os.path.join(
        SAMPLE_DIR,
        filename
    )

    if os.path.exists(direct_path):
        return direct_path

    for root, dirs, files in os.walk(SAMPLE_DIR):

        if filename in files:
            return os.path.join(
                root,
                filename
            )

    return None


# ============================================================
# SAVE GENERATION RESULTS
# ============================================================

def append_generation_result(result):

    exists = os.path.exists(GENERATION_CSV)

    fields = [
        "patent_id",
        "filename",
        "glb_path",
        "ply_path",
        "generation_seconds",
        "peak_vram_gb",
        "status",
        "error"
    ]

    with open(
        GENERATION_CSV,
        "a",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fields
        )

        if not exists:
            writer.writeheader()

        writer.writerow(result)


# ============================================================
# LOAD TRELLIS
# ============================================================

print()
print("=" * 80)
print("Loading TRELLIS")
print("=" * 80)

pipeline = (
    TrellisImageTo3DPipeline
    .from_pretrained(
        "microsoft/TRELLIS-image-large"
    )
)

pipeline.cuda()

print("TRELLIS loaded successfully.")
print()


# ============================================================
# READ IMPACT CSV
# ============================================================

with open(
    CSV_PATH,
    newline="",
    encoding="utf-8"
) as f:

    reader = csv.DictReader(f)

    for row in reader:

        patent_id = row["id"].strip()

        if not patent_id:
            continue

        caption = row.get(
            "caption",
            ""
        ).strip()

        try:

            file_names = ast.literal_eval(
                row["file_names"]
            )

            if isinstance(file_names, str):
                file_names = [file_names]

        except Exception as e:

            print(
                f"Could not parse filenames "
                f"for {patent_id}: {e}"
            )

            continue


        print()
        print("=" * 80)
        print(f"PATENT: {patent_id}")
        print(f"TIFF files: {len(file_names)}")
        print("=" * 80)


        # ----------------------------------------------------
        # Patent output directory
        # ----------------------------------------------------

        patent_output_dir = os.path.join(
            OUTPUT_DIR,
            patent_id
        )

        os.makedirs(
            patent_output_dir,
            exist_ok=True
        )


        # ----------------------------------------------------
        # Save caption
        # ----------------------------------------------------

        caption_path = os.path.join(
            patent_output_dir,
            "caption.txt"
        )

        with open(
            caption_path,
            "w",
            encoding="utf-8"
        ) as caption_file:

            caption_file.write(caption)


        # ----------------------------------------------------
        # Process each TIFF
        # ----------------------------------------------------

        for filename in file_names:

            tiff_path = find_tiff(filename)

            if tiff_path is None:

                print(
                    f"NOT FOUND: {filename}"
                )

                append_generation_result({
                    "patent_id": patent_id,
                    "filename": filename,
                    "glb_path": "",
                    "ply_path": "",
                    "generation_seconds": "",
                    "peak_vram_gb": "",
                    "status": "missing_tiff",
                    "error": "TIFF not found"
                })

                continue


            image_id = os.path.splitext(
                filename
            )[0]


            glb_path = os.path.join(
                patent_output_dir,
                image_id + ".glb"
            )

            ply_path = os.path.join(
                patent_output_dir,
                image_id + ".ply"
            )


            # ------------------------------------------------
            # Resume support
            # ------------------------------------------------

            if os.path.exists(glb_path):

                print(
                    f"SKIPPING existing: {filename}"
                )

                continue


            print()
            print(
                f"Processing: {filename}"
            )


            try:

                image = (
                    Image.open(tiff_path)
                    .convert("RGB")
                )


                # --------------------------------------------
                # GPU metrics
                # --------------------------------------------

                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()

                start_time = (
                    time.perf_counter()
                )


                # --------------------------------------------
                # TRELLIS inference
                # --------------------------------------------

                with torch.no_grad():

                    outputs = pipeline.run(
                        image,
                        seed=1
                    )


                torch.cuda.synchronize()

                generation_seconds = (
                    time.perf_counter()
                    - start_time
                )

                peak_vram_gb = (
                    torch.cuda
                    .max_memory_allocated()
                    / (1024 ** 3)
                )


                # --------------------------------------------
                # Export GLB
                # --------------------------------------------

                glb = (
                    postprocessing_utils
                    .to_glb(
                        outputs["gaussian"][0],
                        outputs["mesh"][0],
                        simplify=0.95,
                        texture_size=1024
                    )
                )

                glb.export(
                    glb_path
                )


                # --------------------------------------------
                # Export Gaussian PLY
                # --------------------------------------------

                outputs[
                    "gaussian"
                ][0].save_ply(
                    ply_path
                )


                print(
                    f"Saved GLB: {glb_path}"
                )

                print(
                    f"Saved PLY: {ply_path}"
                )

                print(
                    f"Generation time: "
                    f"{generation_seconds:.2f} sec"
                )

                print(
                    f"Peak VRAM: "
                    f"{peak_vram_gb:.2f} GB"
                )


                append_generation_result({

                    "patent_id":
                        patent_id,

                    "filename":
                        filename,

                    "glb_path":
                        glb_path,

                    "ply_path":
                        ply_path,

                    "generation_seconds":
                        generation_seconds,

                    "peak_vram_gb":
                        peak_vram_gb,

                    "status":
                        "success",

                    "error":
                        ""

                })


            except Exception as e:

                print()
                print(
                    f"ERROR processing "
                    f"{filename}"
                )

                print(e)

                traceback.print_exc()


                append_generation_result({

                    "patent_id":
                        patent_id,

                    "filename":
                        filename,

                    "glb_path":
                        "",

                    "ply_path":
                        "",

                    "generation_seconds":
                        "",

                    "peak_vram_gb":
                        "",

                    "status":
                        "failed",

                    "error":
                        str(e)

                })


                # Do not kill the entire dataset
                continue


print()
print("=" * 80)
print("ALL GENERATION FINISHED")
print("=" * 80)

print(
    f"Generation statistics: "
    f"{GENERATION_CSV}"
)
