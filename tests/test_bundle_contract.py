import csv, json, tempfile, unittest
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from runtime.bundle_writer import write_run_bundle
from runtime.ground_truth import GroundTruthRecord

class BundleContracts(unittest.TestCase):
    def fake(self):
        tcn=[]
        for e,label in [(15,'Non-Danger'),(23,'Non-Danger'),(31,'Danger')]: tcn.append({'window_start':e-15,'window_end':e,'label':label,'probabilities':[0.8,0.2] if label=='Non-Danger' else [0.1,0.9],'status':'ok'})
        stds=[{'window_start':0,'window_end':63,'pre_crf_label':'Precursor','pre_crf_probabilities':[0.1,0.8,0.1],'post_crf_label':'Normal','post_crf_path':[0]*64,'status':'ok'}]
        dbn=[{'window_start':0,'window_end':63,'label':'Precursor','probabilities':[0.2,0.7,0.1],'t4':0.0,'sp':1.0,'sd':-1.0,'stds_probabilities':[0.1,0.8,0.1],'status':'ok'}]
        return {'source':'fake','frame_count':80,'fps':20.0,'tcn':tcn,'stds':stds,'dbn':dbn,'schema_version':'2.0','status':'ok','model_status':{'tcn':'ok','stds':'ok','dbn':'ok'},'environment':{}}
    def test_bundle_repeat_and_required_files(self):
        root=Path(__file__).resolve().parents[1]
        gt=GroundTruthRecord('ENV::V',0,'ENV','V',__import__('numpy').array([0]*20+[1]*20+[2]*40))
        with tempfile.TemporaryDirectory() as td:
            a=write_run_bundle(root,self.fake(),None,td,'ENV::V','evaluate','research',gt,None,create_overlay=False)
            b=write_run_bundle(root,self.fake(),None,td,'ENV::V','evaluate','product',gt,None,create_overlay=False)
            self.assertTrue(Path(a['run_dir']).name.endswith('_run_001')); self.assertTrue(Path(b['run_dir']).name.endswith('_run_002'))
            for r in (a,b):
                p=Path(r['run_dir']); self.assertEqual(json.loads((p/'_evidence/validation_status.json').read_text(encoding='utf-8'))['status'],'ok')
                for f in ('frame_timeline.csv','transition_summary.csv','metrics_summary.json','result.json','_evidence/run_manifest.json','_evidence/output_hashes.json'): self.assertTrue((p/f).exists(),f)
            prod=json.loads((Path(b['run_dir'])/'result.json').read_text(encoding='utf-8')); self.assertNotIn('stds',prod); self.assertNotIn('t4',prod['dbn'][0])
    def test_gt_length_mismatch_fails(self):
        root=Path(__file__).resolve().parents[1]
        gt=GroundTruthRecord('ENV::V',0,'ENV','V',__import__('numpy').array([0]*3))
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(ValueError): write_run_bundle(root,self.fake(),None,td,'ENV::V','evaluate','research',gt,None)
if __name__=='__main__': unittest.main()
