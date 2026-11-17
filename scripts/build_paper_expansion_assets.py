"""Paper figures and view-subset diagnostics from measured archives and prepared inputs."""
import csv
import statistics
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'Latex/skecth_3d_cvpr27/figures'
PREVIEW=ROOT/'experiments/sketch_adaptation/pilot_preview/D0908314_0000'


def save(fig,name):
    fig.savefig(OUT/(name+'.png'),dpi=300,bbox_inches='tight',pad_inches=.04)
    fig.savefig(OUT/(name+'.svg'),bbox_inches='tight',pad_inches=.04)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.spines.top':False,'axes.spines.right':False})
    fig,ax=plt.subplots(figsize=(7,2.9));ax.set(xlim=(0,14),ylim=(0,5));ax.axis('off')
    def box(x,y,w,h,title,body,color='#eef3f7'):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.06',facecolor=color,edgecolor='#3e596b',linewidth=.8))
        ax.text(x+w/2,y+h-.25,title,ha='center',va='top',fontweight='bold',fontsize=8)
        ax.text(x+w/2,y+h/2-.18,body,ha='center',va='center',fontsize=7.3,linespacing=1.4)
    def arrow(a,b,color='#3e596b',style='-'):
        ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=8,color=color,linewidth=.9,linestyle=style))
    box(.1,2,2,2.5,'Patent drawing','Reviewed crop\nWhite RGB canvas')
    box(2.6,2,3.0,2.5,'Input branches','Stock / standard prep.\nRGB / bypass prep.\nDistance-to-ink map\nDistance map + blur','#eaf3ec')
    box(6.1,2,2.2,2.5,'Frozen TRELLIS','Same checkpoint\nMatched seeds\nNo text conditioning')
    box(8.8,2,2,2.5,'3D asset','Textured mesh\nEight views\nFixed renderer')
    box(11.3,2,2.5,2.5,'Assessment','Image + caption scores\nMesh / failure checks\nPatent aggregation')
    for a,b in [((2.1,3.25),(2.55,3.25)),((5.6,3.25),(6.05,3.25)),((8.3,3.25),(8.75,3.25)),((10.8,3.25),(11.25,3.25))]:arrow(a,b)
    ax.plot([1.1,1.1,12.55],[2,1.05,1.05],color='#6a8a64',linewidth=.9)
    arrow((12.55,1.05),(12.55,1.95),'#6a8a64')
    ax.text(6.7,.8,'Common untransformed reference + patent caption (evaluation only)',ha='center',va='top',fontsize=8,color='#476542')
    fig.tight_layout(pad=.2);save(fig,'workflow')
    fig,axes=plt.subplots(1,3,figsize=(3.3,1.7))
    for ax,name,title in zip(axes,['reference','distance','distance_blur'],['RGB reference','Distance map','Map + smoothing']):
        with Image.open(PREVIEW/(name+'.png')) as image:ax.imshow(image)
        ax.set_title(title,fontsize=7);ax.axis('off')
    fig.tight_layout(pad=.1,w_pad=.25);save(fig,'input_ablation')
    rows=list(csv.DictReader((ROOT/'data/metrics/TRELLIS_Complete_Results/metrics_per_view.csv').open()))
    samples=defaultdict(dict)
    for r in rows:
        key=(r['patent_id'],r['filename'])
        if r['view'] in samples[key]:raise ValueError('Duplicate view')
        samples[key][r['view']]=r
    names=[f'az{i:03d}' for i in range(0,360,45)]
    if not all(set(views)==set(names) for views in samples.values()):raise ValueError('Incomplete view archive')
    subset_specs=[('Fixed',names[:1]),('Opposed',[names[0],names[4]]),('Four',names[::2]),('Eight',names)]
    result=[]
    for label,subset in subset_specs:
        record={'views':len(subset),'selection':label}
        for metric,aggregation in [('ssim','mean'),('ssim','best'),('lpips','mean'),('clip_image_similarity','mean')]:
            patents=defaultdict(list)
            for (patent,_),views in samples.items():
                vals=[float(views[name][metric]) for name in subset]
                value=statistics.mean(vals) if aggregation=='mean' else max(vals)
                patents[patent].append(value)
            means=[statistics.mean(values) for values in patents.values()]
            record[metric+'_'+aggregation]=statistics.mean(means)
        result.append(record)
    with (OUT/'view_subset_metrics.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(result[0]));w.writeheader();w.writerows(result)
    table=[r'\begin{table}[t]',r'\centering\small',r'\caption{View-subset diagnostic on our generated reconstruction set (438 drawings, 50 patents). Subsets are fixed before scoring. Values are patent-level means; best SSIM selects a view separately for each drawing. This changes evaluation coverage, not generation.}',r'\label{tab:view_subsets}',r'\begin{tabular*}{\columnwidth}{@{\extracolsep{\fill}}lrrr@{}}',r'\toprule',r'Views & Mean SSIM & Best SSIM & LPIPS \\',r'\midrule']
    for r in result:table.append(f"{r['views']} & {r['ssim_mean']:.4f} & {r['ssim_best']:.4f} & {r['lpips_mean']:.4f} "+r'\\')
    table += [r'\bottomrule',r'\end{tabular*}',r'\end{table}','']
    (OUT.parent/'sec/view_subset_table.tex').write_text('\n'.join(table))
    fig,axes=plt.subplots(1,2,figsize=(3.3,1.8))
    per_angle=[]
    for metric,ax,title in zip(['ssim','lpips'],axes,['SSIM (higher better)','LPIPS (lower better)']):
        means=[]
        for name in names:
            groups=defaultdict(list)
            for (patent,_),views in samples.items():groups[patent].append(float(views[name][metric]))
            vals=[statistics.mean(values) for values in groups.values()]
            means.append(statistics.mean(vals));per_angle.append(dict(metric=metric,view=name,mean=means[-1],patents=len(vals)))
        ax.plot(range(0,360,45),means,'o-',color='#315f7b',markersize=2.8,linewidth=1)
        ax.set_title(title,fontsize=7);ax.set_xticks([0,90,180,270]);ax.tick_params(labelsize=6)
        ax.set_xlabel('Azimuth (degrees)',fontsize=7);ax.grid(alpha=.2)
    fig.tight_layout(pad=.25,w_pad=.6);save(fig,'view_sensitivity')
    with (OUT/'view_angle_metrics.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(per_angle[0]));w.writeheader();w.writerows(per_angle)
    print(result)


if __name__=='__main__':main()
