"""Inspect copied D0908314 evaluation images without rerunning inference."""
import ast
import csv
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter


def structural_similarity(a,b,channel_axis=2,data_range=255):
    """Match evaluator default SSIM: 7x7 uniform window, sample covariance."""
    a,b=a.astype(float),b.astype(float)
    scores=[]
    for channel in range(3):
        x,y=a[:,:,channel],b[:,:,channel]
        ux,uy=uniform_filter(x,7),uniform_filter(y,7)
        vx=(uniform_filter(x*x,7)-ux*ux)*49/48
        vy=(uniform_filter(y*y,7)-uy*uy)*49/48
        cov=(uniform_filter(x*y,7)-ux*uy)*49/48
        c1,c2=(.01*data_range)**2,(.03*data_range)**2
        score=((2*ux*uy+c1)*(2*cov+c2))/((ux*ux+uy*uy+c1)*(vx+vy+c2))
        scores.append(score[3:-3,3:-3].mean())
    return float(np.mean(scores))

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'experiments/sketch_to_3d/D0908314_evaluation'
OUT=ROOT/'reports/matched_evaluation'


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    # Reuse only the evaluator's pure image functions, avoiding model imports.
    evaluator=ROOT/'experiments/sketch_to_3d/evaluate_trellis_outputs.py'
    tree=ast.parse(evaluator.read_text())
    names={'pil_rgb','foreground_mask','bbox_from_mask','crop_and_center'}
    functions=ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names],type_ignores=[])
    scope={'cv2':cv2,'np':np,'Image':Image}
    exec(compile(functions,str(evaluator),'exec'),scope)
    with (ROOT/'data/metrics/TRELLIS_Complete_Results/metrics_per_view.csv').open() as f:
        rows=[r for r in csv.DictReader(f) if r['patent_id']=='D0908314']
    folders=sorted(p for p in SOURCE.iterdir() if p.is_dir() and (p/'reference_aligned.png').is_file())
    fig,axes=plt.subplots(len(folders),3,figsize=(9,3*len(folders)),squeeze=False)
    maskfig,maskaxes=plt.subplots(len(folders),3,figsize=(9,3*len(folders)),squeeze=False)
    audit=[]
    for i,folder in enumerate(folders):
        candidates=[r for r in rows if Path(r['filename']).stem==folder.name]
        best=max(candidates,key=lambda r:float(r['ssim']))
        ref=Image.open(folder/'reference_aligned.png').convert('RGB')
        rendered=Image.open(folder/(best['view']+'_aligned.png')).convert('RGB')
        fixed=Image.open(folder/'az090_aligned.png').convert('RGB')
        measured=structural_similarity(np.asarray(ref),np.asarray(rendered),channel_axis=2,data_range=255)
        # Raw TIFFs are absent. Re-thresholding the saved reference is diagnostic;
        # it cannot reproduce the pre-resize original mask exactly.
        reference_mask=scope['foreground_mask'](ref)
        render_aligned,render_mask=scope['crop_and_center'](Image.open(folder/(best['view']+'.png')),ref.width)
        equal=np.array_equal(np.asarray(render_aligned),np.asarray(rendered))
        intersection=np.logical_and(reference_mask,render_mask).sum()
        union=np.logical_or(reference_mask,render_mask).sum()
        diagnostic_iou=intersection/union if union else float('nan')
        overlay=np.ones((*reference_mask.shape,3))
        overlay[reference_mask]=[.1,.4,.85]
        overlay[render_mask]=[.95,.45,.1]
        overlay[reference_mask & render_mask]=[.3,.65,.3]
        for ax,img,title in zip(axes[i],[ref,rendered,fixed],[f'Drawing {folder.name[-6:]}\nAligned TIFF reference',f'Best SSIM view: {best["view"]}\nSSIM={measured:.3f}', 'Fixed view: az090']):
            ax.imshow(img);ax.set_title(title,fontsize=10);ax.axis('off')
        for ax,img,title in zip(maskaxes[i],[reference_mask,render_mask,overlay],['Reference mask (re-thresholded)','Render mask (evaluator method)',f'Mask overlay\nDiagnostic IoU={diagnostic_iou:.4f}']):
            ax.imshow(img,cmap='gray',vmin=0,vmax=1);ax.set_title(title,fontsize=10);ax.axis('off')
        audit.append(dict(filename=best['filename'],selected_view=best['view'],recorded_ssim=float(best['ssim']),recomputed_ssim=measured,ssim_abs_difference=abs(measured-float(best['ssim'])),render_alignment_exact_match=equal,reference_foreground_fraction=reference_mask.mean(),render_foreground_fraction=render_mask.mean(),recorded_iou=float(best['silhouette_iou']),diagnostic_rethresholded_iou=diagnostic_iou))
    fig.suptitle('D0908314: all five copied drawings and corresponding TRELLIS renders\nBest-view selection uses recorded SSIM; fixed az090 provides a consistent alternative',fontsize=12)
    maskfig.suptitle('Foreground-mask diagnostic — line drawing versus shaded render\nBlue: reference only; orange: render only; green: overlap. Not filled object silhouettes.',fontsize=12)
    for name,figure in [('matched_comparison',fig),('mask_diagnostic',maskfig)]:
        figure.tight_layout(rect=(0,0,1,.95))
        figure.savefig(OUT/(name+'.png'),dpi=200,bbox_inches='tight')
        figure.savefig(OUT/(name+'.svg'),bbox_inches='tight')
        plt.close(figure)
    with (OUT/'image_checks.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(audit[0]));w.writeheader();w.writerows(audit)
    (OUT/'README.md').write_text('''# Matched evaluation inspection: D0908314

All five copied drawing folders are included, without quality-based sample exclusion. The reference images are the evaluator's saved aligned TIFF derivatives, not raw TIFFs. Each row shows the reference, the render selected by maximum recorded SSIM across eight views, and a fixed az090 render. This is one patent and one imported pipeline, not a representative three-method benchmark.

The mask diagnostic applies the evaluator's actual foreground function to the saved aligned reference. It reprocesses the raw render with the actual crop function. Because original TIFFs are absent, the reference mask is re-thresholded after resize and is not guaranteed to equal the mask used during the original evaluation. Diagnostic IoU is therefore separate from recorded IoU. Image checks include exact render-alignment comparison and recomputed SSIM on the saved images.

The thresholded patent foreground represents ink/shading rather than a filled object silhouette. Thin lines may be removed by 3x3 opening and component filtering; reference and shaded-render masks need visual validation before interpreting IoU or contour distances as object-shape accuracy. Independently cropping each image normalizes size and margins but does not register pose or orientation.

Suggested caption: Matched qualitative examples for patent D0908314. Left: preprocessed patent drawings. Middle: imported TRELLIS renders selected by highest SSIM over eight horizontal viewpoints. Right: the same meshes rendered at a fixed 90-degree azimuth. All five available drawing/model pairs from this patent are shown. Best-view selection is used for illustration and does not demonstrate camera correspondence or performance relative to other methods.

Use the comparison in qualitative results with its selection rule. Keep the mask diagnostic in evaluation validation or the appendix. Inspect examples from additional patents before making general quality claims. To assess filled silhouettes, validate object masks against manually checked references and use renderer depth/alpha for generated foreground in a separate, versioned evaluation; do not silently replace old scores.

Regenerate with scripts/build_matched_evaluation_figure.py (numpy, Pillow, OpenCV, matplotlib, scipy). Sources: copied D0908314_evaluation images, imported metrics_per_view.csv, and evaluate_trellis_outputs.py. No model inference or source-file changes were performed.
''')
    print(f'Inspected {len(audit)} pairs; max SSIM discrepancy={max(r["ssim_abs_difference"] for r in audit):.3g}; all render alignments match={all(r["render_alignment_exact_match"] for r in audit)}')


if __name__=='__main__':
    main()
