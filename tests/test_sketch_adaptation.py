"""CPU controls for line preservation, manifest isolation and paired aggregation."""
import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from PIL import Image
from experiments.sketch_adaptation.run import prepare_images, read_manifest, prepare, generate
from experiments.sketch_adaptation.results import summarize


class AdaptationTests(unittest.TestCase):
    def test_single_pixel_line_is_preserved(self):
        array=np.full((64,64,3),255,dtype=np.uint8);array[:,32]=0
        images,_=prepare_images(Image.fromarray(array),64,2,1)
        self.assertEqual(np.count_nonzero(np.array(images['ink_mask'])),64)
        distance=np.array(images['distance'])[:,:,0]
        self.assertTrue((distance[:,32]==0).all())
        self.assertTrue((distance[:,33]>0).all())
        self.assertTrue((distance[:,0]==255).all())
        self.assertFalse(np.array_equal(images['distance'],images['distance_blur']))

    def test_blank_is_rejected(self):
        with self.assertRaisesRegex(ValueError,'uniform'):
            prepare_images(Image.new('RGB',(64,64),'white'),64,2,1)

    def test_patent_split_leakage_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory);Image.new('RGB',(64,64)).save(path/'source.png')
            rows=[dict(sample_id=str(i),patent_id='P',split=split,image='source.png',crop=[0,0,64,64],crop_reviewed=True) for i,split in enumerate(['dev','test'])]
            manifest=path/'manifest.jsonl';manifest.write_text(''.join(json.dumps(r)+'\n' for r in rows))
            with self.assertRaisesRegex(ValueError,'both dev and test'): read_manifest(manifest)

    def test_prepare_and_dry_run_without_gpu(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            array=np.full((64,80,3),255,dtype=np.uint8);array[10:55,35]=0
            Image.fromarray(array).save(root/'source.png')
            manifest=root/'source.jsonl'
            manifest.write_text(json.dumps(dict(sample_id='one',patent_id='P',split='dev',image='source.png',crop=[0,0,80,64],crop_reviewed=True,caption='Object'))+'\n')
            prepared=root/'prepared'
            prepare(SimpleNamespace(manifest=manifest,output=prepared,size=64,width=2,blur=1))
            common=prepared/'dev/IMPACT/Sample data/one.png'
            self.assertEqual(common.read_bytes(),(prepared/'inputs/one/reference.png').read_bytes())
            with Image.open(common) as image: self.assertEqual(image.size,(64,64))
            trellis_root=root/'trellis';(trellis_root/'trellis/pipelines').mkdir(parents=True)
            (trellis_root/'trellis/pipelines/trellis_image_to_3d.py').write_text('# stub for CPU path validation')
            generate(SimpleNamespace(prepared=prepared,split='dev',seeds=[1],variants=['stock','distance'],trellis_root=trellis_root,model='test',dry_run=True,output=root/'not_created'))
            self.assertFalse((root/'not_created').exists())

    def test_paired_summary_weights_patents_and_reports_failures(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);prepared=root/'prepared';prepared.mkdir();run=root/'run';run.mkdir()
            rows=[dict(patent_id=p,sample_id=s,split='test') for p,s in [('A','a1'),('A','a2'),('B','b1'),('C','failed')]]
            (prepared/'manifest.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows));(prepared/'protocol.json').write_text('{}')
            sha=lambda name:hashlib.sha256((prepared/name).read_bytes()).hexdigest()
            config=dict(prepared=str(prepared),split='test',variants=['stock','distance'],seeds=[1,2],manifest_sha256=sha('manifest.jsonl'),protocol_sha256=sha('protocol.json'))
            (run/'run.json').write_text(json.dumps(config))
            for variant in config['variants']:
                for seed in config['seeds']:
                    folder=run/variant/f'seed-{seed}';folder.mkdir(parents=True)
                    with (folder/'metrics_summary.csv').open('w',newline='') as f:
                        writer=csv.DictWriter(f,fieldnames=['patent_id','filename','status','mean_ssim']);writer.writeheader()
                        for p,s,value in [('A','a1',0),('A','a2',0),('B','b1',1)]:
                            writer.writerow(dict(patent_id=p,filename=s+'.png',status='success',mean_ssim=value+(.1 if variant=='distance' else 0)))
                            (folder/p).mkdir(exist_ok=True)
                            (folder/p/(s+'.run.json')).write_text(json.dumps({'status':'success'}))
            summarize(SimpleNamespace(run=run))
            with (run/'analysis/paired_metrics.csv').open() as f:
                summary=list(csv.DictReader(f))
            stock=next(r for r in summary if r['metric']=='mean_ssim' and r['variant']=='stock')
            adapted=next(r for r in summary if r['metric']=='mean_ssim' and r['variant']=='distance')
            self.assertAlmostEqual(float(stock['mean']),.5)  # equal patents, not .333 image mean
            self.assertAlmostEqual(float(adapted['delta_vs_stock']),.1)
            self.assertEqual(int(stock['matched_images']),3)
            with (run/'analysis/coverage.csv').open() as f:
                coverage=list(csv.DictReader(f))
            self.assertTrue(all(int(r['not_successfully_evaluated'])==1 for r in coverage))


if __name__=='__main__':unittest.main()
