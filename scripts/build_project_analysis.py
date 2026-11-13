"""Reproducible archive analysis and geometry-only inspection figures.
Requires numpy and matplotlib. Does not evaluate reconstruction fidelity.
"""
import csv
import json
import struct
from pathlib import Path
from collections import Counter
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'data/metrics/TRELLIS_Complete_Results'
OUT = ROOT / 'reports/project_analysis'
QUALITY = [('mean_ssim', 'SSIM ↑'), ('mean_psnr', 'PSNR (dB) ↑'),
           ('mean_lpips', 'LPIPS ↓'), ('mean_silhouette_iou', 'Silhouette IoU ↑'),
           ('mean_contour_chamfer_norm', 'Contour Chamfer ↓'),
           ('mean_clip_image_similarity', 'CLIP image ↑'),
           ('mean_clip_caption_similarity', 'CLIP caption ↑')]


def read(path):
    with path.open(newline='') as f:
        return list(csv.DictReader(f))


def values(rows, key):
    return np.array([float(r[key]) for r in rows if r.get(key) and np.isfinite(float(r[key]))])


def write_csv(name, rows):
    with (OUT / (name + '.csv')).open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def table(name, title, headers, rows, note):
    (OUT / (name + '.md')).write_text('\n'.join([
        '# ' + title, '', '| ' + ' | '.join(headers) + ' |',
        '| ' + ' | '.join(['---'] * len(headers)) + ' |',
        *['| ' + ' | '.join(map(str, r)) + ' |' for r in rows], '', note, '']))
    esc = lambda s: str(s).replace('%', r'\%').replace('_', r'\_').replace('±', r'$\pm$')
    (OUT / (name + '.tex')).write_text('\n'.join([
        '% Requires booktabs. ' + note,
        r'\begin{table*}[t]', r'\centering\small', r'\caption{' + esc(title + '. ' + note) + '}',
        r'\begin{tabular}{' + 'l' * len(headers) + '}', r'\toprule',
        ' & '.join(map(esc, headers)) + r' \\', r'\midrule',
        *[' & '.join(map(esc, r)) + r' \\' for r in rows],
        r'\bottomrule', r'\end{tabular}', r'\end{table*}', '']))


def geometry(path):
    """Read indexed triangle geometry from the simple imported GLB format.
    Reject transforms/multiple primitives instead of silently changing semantics.
    """
    b = path.read_bytes()
    magic, version, length = struct.unpack_from('<4sII', b)
    if magic != b'glTF' or version != 2 or length != len(b):
        raise ValueError('Invalid GLB header')
    size, kind = struct.unpack_from('<II', b, 12)
    if kind != 0x4e4f534a:
        raise ValueError('Missing JSON chunk')
    j = json.loads(b[20:20+size])
    bin_size, bin_kind = struct.unpack_from('<II', b, 20+size)
    if bin_kind != 0x004e4942 or len(j.get('buffers', [])) != 1 or 'uri' in j['buffers'][0]:
        raise ValueError('Expected embedded single buffer')
    if len(j.get('nodes', [])) != 1 or any(k in j['nodes'][0] for k in ['matrix','translation','rotation','scale','children']):
        raise ValueError('Unsupported node hierarchy/transform')
    if len(j['meshes']) != 1 or len(j['meshes'][0]['primitives']) != 1:
        raise ValueError('Expected single primitive')
    p = j['meshes'][0]['primitives'][0]
    if p.get('mode', 4) != 4 or 'extensions' in p:
        raise ValueError('Unsupported primitive')
    raw = memoryview(b)[28+size:28+size+bin_size]
    def accessor(i):
        a = j['accessors'][i]
        if 'sparse' in a:
            raise ValueError('Sparse accessor unsupported')
        v = j['bufferViews'][a['bufferView']]
        dtype = {5126:'<f4',5125:'<u4',5123:'<u2',5121:'u1'}[a['componentType']]
        width = {'SCALAR':1,'VEC3':3}[a['type']]
        item = np.dtype(dtype).itemsize
        offset = v.get('byteOffset',0) + a.get('byteOffset',0)
        stride = v.get('byteStride', width*item)
        return np.ndarray((a['count'],width),dtype=dtype,buffer=raw,offset=offset,strides=(stride,item)).copy()
    vertices = accessor(p['attributes']['POSITION'])
    faces = accessor(p['indices']).reshape(-1,3).astype(np.int64)
    if not np.isfinite(vertices).all() or faces.min() < 0 or faces.max() >= len(vertices):
        raise ValueError('Invalid geometry values')
    return vertices, faces


def topology(v, f):
    edges = np.sort(np.vstack([f[:,[0,1]],f[:,[1,2]],f[:,[2,0]]]),axis=1)
    unique, counts = np.unique(edges,axis=0,return_counts=True)
    used = np.unique(f)
    parent = list(range(len(v)))
    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for a, b in unique:
        a, b = root(int(a)), root(int(b))
        if a != b:
            parent[b] = a
    labels = np.array([root(i) for i in range(len(v))])
    labels_used = labels[used]
    component_count = len(np.unique(labels_used))
    # Component colors reflect index connectivity; no position welding is applied.
    cross = np.cross(v[f[:,1]]-v[f[:,0]], v[f[:,2]]-v[f[:,0]])
    return dict(vertices=len(v), faces=len(f), connected_components=component_count,
                boundary_edges=int((counts==1).sum()), nonmanifold_edges=int((counts>2).sum()),
                degenerate_faces=int((np.linalg.norm(cross,axis=1) <= 1e-12).sum()),
                watertight_edges=bool(np.all(counts==2)),
                largest_component_vertex_fraction=float(np.bincount(labels_used).max()/len(used))), labels


def savefig(name, fig):
    fig.savefig(OUT / (name+'.png'),dpi=220,bbox_inches='tight')
    fig.savefig(OUT / (name+'.svg'),bbox_inches='tight')
    plt.close(fig)


def mesh_panel(ax, v, f, azimuth, labels=None):
    center = (v.max(axis=0)+v.min(axis=0))/2
    scale = max(np.ptp(v,axis=0))
    v = (v-center)/scale
    colors = '#7399ad'
    if labels is not None:
        face_labels = labels[f[:,0]]
        colors = plt.get_cmap('tab20')((face_labels % 20)/19)
    # Directional face lighting makes shape visible without applying textures.
    from matplotlib.colors import to_rgba
    normals = np.cross(v[f[:,1]]-v[f[:,0]], v[f[:,2]]-v[f[:,0]])
    norm = np.linalg.norm(normals,axis=1)
    normals = normals / np.maximum(norm[:,None],1e-15)
    light = np.array([.4,-.6,.7]); light /= np.linalg.norm(light)
    intensity = .35 + .65*np.abs(normals @ light)
    colors = np.tile(to_rgba(colors),(len(f),1)) if isinstance(colors,str) else colors.copy()
    colors[:,:3] *= intensity[:,None]
    ax.add_collection3d(Poly3DCollection(v[f],facecolors=colors,edgecolors='none',linewidths=0,rasterized=True))
    ax.set(xlim=(-.55,.55),ylim=(-.55,.55),zlim=(-.55,.55))
    ax.set_box_aspect((1,1,1)); ax.view_init(elev=20,azim=azimuth); ax.set_axis_off()


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    imported = read(SOURCE/'metrics_summary.csv')
    new = read(ROOT/'data/metrics/metrics_new_pipeline.csv')
    know = read(ROOT/'data/metrics/metrics_know3d_pipeline.csv')
    tex = read(ROOT/'data/metrics/texverse_evaluation_results.csv')
    generation = read(SOURCE/'generation_metrics.csv')
    counts = Counter(r['patent_id'] for r in imported)
    inputs = sum(Path(r['tiff_path']).is_file() for r in imported)
    matched = sum((SOURCE/r['patent_id']/(Path(r['filename']).stem+'.glb')).is_file() for r in imported)
    dataset = [['Imported patent archive',len(counts),len(imported),matched,'Unverified'],
               ['New Pipeline CSV',len({r['patent_id'] for r in new}),len(new),'Not independent','Adjusted'],
               ['Know3D CSV',len({r['patent_id'] for r in know}),len(know),'Not independent','Copied/scaled'],
               ['TexVerse CSV','Unknown',len(tex),'Unavailable','Synthetic']]
    note = f'Counts describe archived records, not verified dataset coverage. Original TIFF paths accessible: {inputs}/{len(imported)}. Generation log rows: {len(generation)}. No cross-dataset pairing is assumed.'
    table('dataset_summary','Dataset and artifact inventory',['Archive','Patents','Rows','Matched GLBs','Provenance'],dataset,note)
    write_csv('images_per_patent',[{'patent_id':k,'images':v} for k,v in sorted(counts.items())])
    completeness=[]
    for name, rows, keys in [('Imported',imported,[k for k,_ in QUALITY]),('New Pipeline',new,[k for k,_ in QUALITY]),('Know3D',know,[k for k,_ in QUALITY]),('TexVerse',tex,[k for k in tex[0] if k!='sample_id'])]:
        for k in keys:
            valid=len(values(rows,k))
            completeness.append(dict(archive=name,metric=k,total=len(rows),valid=valid,missing_or_nonfinite=len(rows)-valid))
    write_csv('metric_completeness',completeness)
    table('metric_completeness','Metric completeness',['Archive','Metric','Total','Valid','Missing'],[list(r.values()) for r in completeness],'Blank and non-finite scores count as missing; zeros remain valid observations.')
    # Imported scores only: avoid comparisons that duplicate or adjust baseline values.
    fig, axes=plt.subplots(2,4,figsize=(14,7))
    rng=np.random.default_rng(42)
    for ax,(key,label) in zip(axes.flat,QUALITY):
        x=values(imported,key)
        ax.boxplot(x,positions=[1],widths=.3,showfliers=False)
        ax.scatter(1+rng.uniform(-.16,.16,len(x)),x,s=8,alpha=.25,color='#245a76',rasterized=True)
        ax.set_title(f'{label}\nn={len(x)}');ax.set_xticks([])
    axes.flat[-1].axis('off')
    fig.suptitle('Imported archive: recorded image-level metric distributions\nEvaluation implementation and input alignment unverified',fontsize=13)
    fig.tight_layout(rect=(0,0,1,.9));savefig('metric_distributions',fig)
    fig,ax=plt.subplots(figsize=(8,3.5));ax.hist(list(counts.values()),bins=np.arange(min(counts.values())-.5,max(counts.values())+1.5),color='#7399ad',edgecolor='white')
    ax.set(xlabel='Images per patent',ylabel='Number of patents',title='Imported archive sampling');fig.tight_layout();savefig('images_per_patent',fig)
    mesh_rows=[];errors=[]
    for row in imported:
        path=SOURCE/row['patent_id']/(Path(row['filename']).stem+'.glb')
        try:
            v,f=geometry(path);stats,_=topology(v,f)
            mesh_rows.append(dict(patent_id=row['patent_id'],filename=row['filename'],**stats))
        except Exception as exc:
            errors.append(dict(path=str(path),error=str(exc)))
    (OUT/'mesh_errors.json').write_text(json.dumps(errors,indent=2)+'\n')
    if not mesh_rows:
        raise RuntimeError('No meshes analyzed; see mesh_errors.json')
    write_csv('mesh_quality_per_model',mesh_rows)
    mesh_table=[]
    for key,label in [('vertices','Vertices'),('faces','Faces'),('connected_components','Connected components'),('boundary_edges','Boundary edges'),('nonmanifold_edges','Nonmanifold edges'),('degenerate_faces','Degenerate faces'),('largest_component_vertex_fraction','Largest component vertex fraction')]:
        x=np.array([r[key] for r in mesh_rows]);q=np.quantile(x,[.25,.5,.75])
        mesh_table.append([label,*[f'{z:.4f}' if 'fraction' in key else f'{z:,.1f}' for z in [q[1],q[0],q[2]]],str(len(x))])
    mesh_table.append(['Closed-edge meshes (%)',f"{100*np.mean([r['watertight_edges'] for r in mesh_rows]):.2f}",'—','—',str(len(mesh_rows))])
    table('mesh_quality','Recomputed imported mesh topology',['Property','Median / percent','Q1','Q3','n'],mesh_table,
          f'Recomputed from GLB triangle indices; no vertex welding. Closed-edge means every undirected edge has two incident faces, not a full manifold or self-intersection test. Degenerate triangle cross-product norm threshold: 1e-12 in native coordinates. Parse failures: {len(errors)}.')
    # Deterministic selection includes a typical connectivity case and extremes.
    ordered=sorted(mesh_rows,key=lambda r:(r['connected_components'],r['patent_id'],r['filename']))
    selected=[('Fewest components',ordered[0]),('Median component count',ordered[len(ordered)//2]),('Most components',ordered[-1])]
    manifest=[]
    fig,axes=plt.subplots(3,3,figsize=(10,10),subplot_kw={'projection':'3d'})
    for i,(selection,row) in enumerate(selected):
        v,f=geometry(SOURCE/row['patent_id']/(Path(row['filename']).stem+'.glb'))
        for j,az in enumerate([0,90,180]):
            mesh_panel(axes[i,j],v,f,az)
            axes[i,j].set_title(f'{selection if j==0 else ""}\n{row["patent_id"]}, azimuth {az}°',fontsize=9)
        manifest.append(dict(selection=selection,**row))
    fig.suptitle('Geometry-only multiview inspection of imported meshes\nCanonical axes; no original sketch or texture comparison',fontsize=13)
    fig.tight_layout(rect=(0,0,1,.92));savefig('qualitative_mesh_grid',fig)
    # Inspection candidates, not verified semantic failures.
    worst_boundary=max(mesh_rows,key=lambda r:(r['boundary_edges']/r['faces'],r['filename']))
    worst_fragment=ordered[-1]
    fig,axes=plt.subplots(1,2,figsize=(10,5),subplot_kw={'projection':'3d'})
    for ax,(reason,row) in zip(axes,[('Most connected components',worst_fragment),('Largest boundary-edge / face ratio',worst_boundary)]):
        v,f=geometry(SOURCE/row['patent_id']/(Path(row['filename']).stem+'.glb'));stats,labels=topology(v,f)
        mesh_panel(ax,v,f,45,labels)
        ax.set_title(f'{reason}\n{row["patent_id"]}: {stats["connected_components"]:,} components\n{stats["boundary_edges"]:,} boundary edges',fontsize=10)
        manifest.append(dict(selection=reason,**row))
    fig.suptitle('Topology inspection candidates — colors indicate index-connected components\nColors repeat after 20 components; semantic failures cannot be verified without sketches',fontsize=11)
    fig.tight_layout(rect=(0,0,1,.86));savefig('topology_failure_candidates',fig)
    write_csv('figure_selection',manifest)
    (OUT/'README.md').write_text('''# Project analysis outputs

Regenerate with `python3 scripts/build_project_analysis.py` (numpy, matplotlib).

## Tables

- dataset_summary.md / .tex: archive inventory and provenance.
- metric_completeness.md / .tex / .csv: valid and missing counts.
- mesh_quality.md / .tex: medians and quartiles from independently recomputed GLB topology.
- mesh_quality_per_model.csv: all recomputed model measurements.
- images_per_patent.csv: dataset sampling counts.

## Figures (PNG and SVG)

- metric_distributions: recorded imported metrics only; box plots with all observations. Boxes show Q1–Q3, center is median, whiskers extend to observations within 1.5 IQR. Points are deterministically jittered. Scores are not newly evaluated.
- images_per_patent: archive sampling distribution.
- qualitative_mesh_grid: deterministic minimum, median-ranked and maximum component-count meshes, each at three canonical azimuths. Untextured geometry views, not photometric evaluation renders. These are not evidence of sketch fidelity.
- topology_failure_candidates: fragmentation and open-boundary inspection candidates. Component colors repeat; disconnected parts may be intentional. Selection rules and exact filenames are in figure_selection.csv.

## Limitations and paper placement

Dataset inventory belongs with experimental setup. Recomputed topology can be reported as an imported-artifact analysis with index-connectivity definitions; native vertices are not welded. UV seams or duplicated positions can inflate index-connected component and boundary counts; these are not automatically physical cracks. Metric distributions and completeness belong in an archive audit appendix until original evaluation provenance is established. Synthetic/adjusted pipeline scores are excluded from these plots.

No source patent TIFFs are available at their recorded paths. Therefore an original-sketch/mesh comparison and verified missing-part, depth-error or silhouette-failure figure cannot be produced. The geometry grid and topology candidate figure are the available substitutes, explicitly labeled as inspection. To finish those comparisons, supply matched original sketches and camera/alignment information. Source assets under upstream dependency packages are unrelated to these patents.

Imported metric provenance is unverified; new-pipeline values were adjusted by scripts/run_vista_pipeline.py, Know3D copies/scales them, and TexVerse is simulated. Counts of processing success are not reconstruction-quality success rates. No new generation, image similarity evaluation, or significance testing was performed.

Sources: data/metrics/TRELLIS_Complete_Results/metrics_summary.csv, generation_metrics.csv and GLB files; data/metrics/metrics_new_pipeline.csv; metrics_know3d_pipeline.csv; texverse_evaluation_results.csv. Existing inputs and tables remain unchanged. LaTeX files require booktabs and have not been compiled against the paper template.
''')
    print(f'Analyzed {len(mesh_rows)} meshes; {len(errors)} errors. Outputs: {OUT}')


if __name__=='__main__':
    main()
