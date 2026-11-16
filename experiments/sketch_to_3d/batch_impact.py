import os
import csv
import ast
import torch
from PIL import Image

from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils import postprocessing_utils


IMPACT_DIR = "/work/11649/schittala/vista/IMPACT"
CSV_PATH = os.path.join(IMPACT_DIR, "Sample data", "sample_data.csv")
OUTPUT_DIR = "/work/11649/schittala/vista/impact_3d_outputs"

os.environ["SPCONV_ALGO"] = "native"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def find_tiff(filename):
    """
    Find a TIFF anywhere under IMPACT/Sample data.
    """
    sample_dir = os.path.join(IMPACT_DIR, "Sample data")

    for root, dirs, files in os.walk(sample_dir):
        if filename in files:
            return os.path.join(root, filename)

    return None


print("Loading TRELLIS...")

pipeline = TrellisImageTo3DPipeline.from_pretrained(
    "microsoft/TRELLIS-image-large"
)

pipeline.cuda()

print("TRELLIS loaded.")


with open(CSV_PATH, newline="", encoding="utf-8") as f:

    reader = csv.DictReader(f)

    for row_index, row in enumerate(reader):
        if row["id"].strip() != "D0908314":
            continue

        patent_id = row["id"].strip()
        caption = row["caption"].strip()

        try:
            file_names = ast.literal_eval(row["file_names"])
        except Exception as e:
            print(f"Could not parse filenames for {patent_id}: {e}")
            continue

        print()
        print("=" * 80)
        print(f"Patent: {patent_id}")
        print(f"Images: {len(file_names)}")
        print(f"Caption: {caption}")
        print("=" * 80)

        patent_output_dir = os.path.join(
            OUTPUT_DIR,
            patent_id
        )

        os.makedirs(patent_output_dir, exist_ok=True)

        # Save the patent-level caption once
        with open(
            os.path.join(patent_output_dir, "caption.txt"),
            "w",
            encoding="utf-8"
        ) as caption_file:

            caption_file.write(caption)

        for filename in file_names:

            tiff_path = find_tiff(filename)

            if tiff_path is None:
                print(f"NOT FOUND: {filename}")
                continue

            image_id = os.path.splitext(filename)[0]

            print()
            print(f"Processing: {filename}")

            image = Image.open(tiff_path).convert("RGB")

            try:

                with torch.no_grad():

                    outputs = pipeline.run(
                        image,
                        seed=1
                    )

                # Export GLB
                glb = postprocessing_utils.to_glb(
                    outputs["gaussian"][0],
                    outputs["mesh"][0],
                    simplify=0.95,
                    texture_size=1024,
                )

                glb_path = os.path.join(
                    patent_output_dir,
                    image_id + ".glb"
                )

                glb.export(glb_path)

                # Export Gaussian PLY
                ply_path = os.path.join(
                    patent_output_dir,
                    image_id + ".ply"
                )

                outputs["gaussian"][0].save_ply(ply_path)

                print(f"Saved: {glb_path}")
                print(f"Saved: {ply_path}")

            except Exception as e:

                print(f"ERROR processing {filename}")
                print(e)

                # Continue with next image instead of killing batch
                continue


print()
print("Finished.")

