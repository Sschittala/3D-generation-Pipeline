# Manuscript revision for the current evidence

These are replacement passages for the Overleaf manuscript, not edits to its main source. The project currently contains only table fragments. The study below is framed as an evaluation of imported pretrained TRELLIS outputs, not a new generation architecture or a three-method benchmark. Use the existing bibliography key `xiang2025trellis`.

## Suggested title

```latex
\title{Evaluating Image-Conditioned 3D Generation from Patent Drawings}
```

## Abstract

```latex
\begin{abstract}
Patent drawings provide structural cues for three-dimensional
objects but differ substantially from shaded images in their
use of sparse contours, annotations, and multiple depicted
views. We study image-conditioned 3D generation in this setting
through an analysis of imported TRELLIS reconstructions for
438 drawings from 50 patents. Our evaluation combines
multi-view drawing-to-render comparisons with analysis of
exported mesh structure. To account for unequal numbers of
drawings per patent, we average view-level scores within
drawings and drawing-level scores within patents. Qualitative
inspection and checks on saved evaluation images identify
limitations of foreground thresholding and the treatment of
multiple figures within a single image. These observations
motivate separating semantic similarity, image appearance,
and mesh connectivity when assessing patent-derived assets.
The study characterizes an existing reconstruction pipeline
and its evaluation limitations rather than establishing
improvements over independently evaluated competing methods.
\end{abstract}
```

## Introduction

```latex
\section{Introduction}
\label{sec:introduction}

Patent drawings communicate object structure through contours,
sectional views, dashed boundaries, and annotations. Converting
these drawings into 3D assets could support interactive
inspection and exploration of design records. However, a
single drawing leaves depth and occluded surfaces ambiguous,
and a patent image may contain several distinct views rather
than a single isolated object.

Pretrained image-conditioned generators such as
TRELLIS~\cite{xiang2025trellis} provide a practical starting
point for producing 3D assets from visual inputs. Their use
with patent drawings raises two separate questions: what
structures are produced, and how should the resulting assets
be evaluated? A plausible shaded render need not reproduce
the geometry of a particular design. Similarly, similarity
between a line drawing and a rendered image depends on
viewpoint, framing, background, and depiction style as well
as object structure.

We examine imported TRELLIS outputs associated with 438
drawings from 50 patents. The generation scripts process each
TIFF independently using a pretrained image-conditioned
checkpoint. We analyze recorded multi-view evaluation results
and recompute mesh topology statistics from the exported GLBs.
Our principal numerical summaries assign equal weight to each
patent, avoiding disproportionate influence from patents with
many drawings.

This study makes three contributions: (1) an analysis of
imported TRELLIS reconstructions associated with patent
drawings, combining image-based scores and mesh statistics;
(2) a reproducible aggregation procedure that distinguishes
view-level, drawing-level, and patent-level results; and
(3) an inspection of evaluation limitations, including
foreground-mask extraction and multi-figure reference images.
We treat qualitative examples and diagnostic measurements as
complementary evidence rather than interpreting any individual
similarity metric as proof of geometric correctness.
```

## Shortened methodology for the main paper

```latex
\section{Methodology}
\label{sec:methodology}

\subsection{Image-Conditioned Reconstruction}
Patent identifiers, drawing filenames, and captions are read
from the IMPACT sample metadata. Each TIFF is located and
converted to RGB, then independently supplied to the
\texttt{microsoft/TRELLIS-image-large} pipeline with random
seed 1. The pretrained model is executed on CUDA with gradient
computation disabled and without additional training. Standard
pipeline preprocessing remains enabled. Captions are saved as
metadata and used for semantic evaluation; they do not
condition generation. Multiple drawings from the same patent
are not jointly supplied to the model.

The first generated mesh and Gaussian representation are
exported as a textured GLB using the TRELLIS postprocessing
utility with \texttt{simplify=0.95} and
\texttt{texture\_size=1024}. The Gaussian representation is
also saved as PLY. Outputs preserve the source drawing's
filename stem, allowing association with the reference TIFF.

\subsection{Rendering and Image Comparison}
The evaluator centers each reconstruction and uniformly scales
its longest bounding-box dimension to two units. Eight views
are rendered at azimuths $0^{\circ},45^{\circ},\ldots,315^{\circ}$
and elevation $10^{\circ}$. The perspective camera uses a
$40^{\circ}$ vertical field of view and a distance of 3.2
normalized units. The renderer uses a white background,
ambient illumination, and three directional lights; its
default image size is $512\times512$ pixels.

Reference drawings and renders are independently cropped and
centered using threshold-derived foreground bounds. This
normalizes framing but does not register camera pose or
separate individual figures within a reference image. We
compute SSIM, PSNR, AlexNet-based LPIPS, and normalized CLIP
cosine similarities. The default CLIP model is
\texttt{openai/clip-vit-base-patch32}. Image similarity compares
drawings with renders, while text similarity compares patent
captions with renders. These scores characterize image and
semantic agreement, not direct 3D geometric accuracy.
Silhouette measurements are withheld from the main results
pending validation of the foreground masks.

\subsection{Aggregation and Mesh Analysis}
For each metric, valid view scores are averaged within a
drawing and drawing-level scores are averaged within a patent.
For patent $p$, this gives
\[
 M_p=\frac{1}{|\mathcal I_p|}
 \sum_{i\in\mathcal I_p}
 \left(\frac{1}{|\mathcal V_i|}
 \sum_{v\in\mathcal V_i}m_{i,v}\right),
\]
where $\mathcal V_i$ and $\mathcal I_p$ contain valid scores
for the metric being summarized. We report the mean and sample
standard deviation of $M_p$ across patents. Missing values
are excluded rather than replaced with zero. Best-view scores
are retained separately and are not substituted for view means.

Mesh analysis includes vertex and face counts, connectivity,
and boundary and nonmanifold edges. Recomputed connectivity
uses stored triangle indices without welding coincident
vertices. Texture seams and duplicated positions can therefore
influence component and boundary counts; these measurements
are not automatically evidence of physical cracks.
```

## Appendix material

Insert after the manuscript's existing `\appendix` command; do not add a second appendix command if one already exists.

```latex
\section{Implementation and Resource Measurements}
\label{app:implementation}

\paragraph{Evaluation preprocessing.}
The evaluator applies Otsu thresholding to grayscale images,
selects threshold polarity using border pixels, and applies
opening and closing with a $3\times3$ kernel. Components
smaller than $\max(16,\lfloor0.0002HW\rfloor)$ pixels are
removed. The foreground box is expanded by eight percent of
its width and height, clipped to the image boundary, resized
with preserved aspect ratio, and centered on a white square.
Masks use nearest-neighbor resizing. An image without a
detected foreground is resized directly. This mask describes
thresholded appearance rather than a validated filled object
silhouette.

\paragraph{Generation timing.}
The batch script synchronizes CUDA immediately before and
after the pipeline call. Recorded time includes internal
pipeline preprocessing and decoding but excludes initial model
loading, TIFF loading, and subsequent GLB and PLY export.
Peak PyTorch-allocated memory is recorded after resetting peak
statistics and divided by $1024^3$, giving GiB. This includes
allocated model memory and is not total device memory use.
The script skips existing GLBs without appending new timing
records; logged timings therefore do not cover every archived
asset. The archive contains 433 generation records for 438
reconstructions.

\paragraph{Local validation.}
For five drawing/model pairs from patent D0908314, selected-view
SSIM recomputed from saved aligned images matched the recorded
scores to numerical precision. Reapplying the evaluator's crop
procedure to the corresponding raw renders reproduced the saved
aligned renders exactly. These checks validate those image
calculations, not generation provenance or geometric accuracy.
Re-thresholding saved aligned references yielded empty masks
for four pairs; these diagnostic masks are not identical by
construction to the original masks obtained before resizing.
Original TIFFs are required to validate the original masks.
```

## Integration checklist

- Replace the abstract's simulated improvement numbers and unsupported custom-generation claims.
- Retain the original-metrics TRELLIS table under Quantitative Results. Do not reuse adjusted New Pipeline columns as independently measured scores.
- Keep Know3D and SketchDream as related methods unless independent runs are available.
- Add the actual TRELLIS citation to the bibliography. Add appropriate dataset and metric citations after checking the manuscript bibliography.
- Replace all CVPR template instructions and editor contact paragraphs.
- Confirm generation and evaluation software revisions and actual runtime options on Vista. Defaults in source code do not independently prove historical runtime settings.
- Do not describe the current qualitative figure as typical across the dataset: it covers one patent only.
- Recheck page layout and references in the full CVPR project. This file is not a compiled manuscript.
