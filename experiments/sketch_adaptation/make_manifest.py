"""Create a reviewable full-image crop manifest from IMPACT metadata (no inference)."""
import argparse
import ast
import csv
import json
from pathlib import Path
from PIL import Image


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--impact-dir',type=Path,required=True);p.add_argument('--patent-ids',nargs='+',required=True)
    p.add_argument('--split',choices=['dev','test'],required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists(): p.error('Output exists; preserve the existing manifest')
    root=args.impact_dir/'Sample data'
    with (root/'sample_data.csv').open() as f: metadata={r['id'].strip():r for r in csv.DictReader(f)}
    rows=[]
    for patent in args.patent_ids:
        row=metadata[patent];files=ast.literal_eval(row['file_names'])
        if isinstance(files,str):files=[files]
        for i,name in enumerate(files):
            matches=list(root.rglob(name))
            if len(matches)!=1: raise ValueError(f'Expected exactly one TIFF for {name}; found {len(matches)}')
            with Image.open(matches[0]) as image: width,height=image.size
            rows.append(dict(sample_id=f'{patent}_{i:04d}',patent_id=patent,image=str(matches[0].resolve()),
                             crop=[0,0,width,height],crop_reviewed=False,split=args.split,caption=row.get('caption','').strip()))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    print(f'Created {len(rows)} entries. Inspect crops, edit crop boxes, and set crop_reviewed=true before prepare.')


if __name__=='__main__': main()
