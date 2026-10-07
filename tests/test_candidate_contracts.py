import json, tempfile, unittest
from pathlib import Path
import numpy as np
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from runtime.engine import FallRiskInferenceEngine
from runtime.evaluation import make_timeline
from runtime.output_bundle import allocate_run_dir
from runtime.profile import result_for_profile
from runtime.test_guard import enforce_test_protection, split_of_video
from runtime.ground_truth import load_ground_truth, load_video_skeleton
from dd_r01 import infer_skeleton3d

class CandidateContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.eng=FallRiskInferenceEngine(ROOT,device='cpu')

    def test_length_status_contract(self):
        cases=[(0,'no_frames','insufficient_length','insufficient_length'),(15,'insufficient_length','insufficient_length','insufficient_length'),(16,'partial','ok','insufficient_length'),(63,'partial','ok','insufficient_length'),(64,'ok','ok','ok')]
        for n,top,tcn,dbn in cases:
            r=self.eng.infer_skeleton_sequence(np.zeros((n,17,3),np.float32),source=f'n={n}')
            self.assertEqual(r.status,top); self.assertEqual(r.model_status['tcn'],tcn); self.assertEqual(r.model_status['dbn'],dbn)

    def test_stds_pre_post_and_dbn_present(self):
        r=self.eng.infer_skeleton_sequence(np.zeros((64,17,3),np.float32),source='64').to_dict()
        self.assertEqual(len(r['tcn']),7); self.assertEqual(len(r['stds']),1); self.assertEqual(len(r['dbn']),1)
        s=r['stds'][0]
        self.assertEqual(len(s['pre_crf_probabilities']),3); self.assertEqual(len(s['post_crf_path']),64)
        self.assertIn(s['pre_crf_label'],('Normal','Precursor','Danger')); self.assertIn(s['post_crf_label'],('Normal','Precursor','Danger'))

    def test_warmup_and_hold_last(self):
        r=self.eng.infer_skeleton_sequence(np.zeros((80,17,3),np.float32),source='80').to_dict(); tl=make_timeline(r)
        self.assertIsNone(tl[14]['tcn_label']); self.assertIsNotNone(tl[15]['tcn_label'])
        self.assertIsNone(tl[62]['dbn_label']); self.assertIsNotNone(tl[63]['dbn_label'])
        self.assertEqual(tl[63]['dbn_decision_frame'],63); self.assertEqual(tl[78]['dbn_decision_frame'],63); self.assertEqual(tl[79]['dbn_decision_frame'],79)

    def test_product_profile_hides_stds_details(self):
        r=self.eng.infer_skeleton_sequence(np.zeros((64,17,3),np.float32)).to_dict(); p=result_for_profile(r,'product')
        self.assertNotIn('stds',p)
        self.assertNotIn('stds_probabilities',p['dbn'][0]); self.assertNotIn('t4',p['dbn'][0])
        self.assertIn('tcn',p); self.assertIn('dbn',p)

    def test_repeat_run_directory_increments(self):
        with tempfile.TemporaryDirectory() as td:
            a=allocate_run_dir(td,'ENV::Video (2)'); b=allocate_run_dir(td,'ENV::Video (2)')
            self.assertTrue(a.name.endswith('_run_001')); self.assertTrue(b.name.endswith('_run_002'))

    def test_test_guard(self):
        split=json.loads((ROOT/'reference/repeat01_config/split.json').read_text(encoding='utf-8'))
        test=split['test_videos'][0]; val=split['val_videos'][0]
        self.assertEqual(split_of_video(ROOT,test),'test'); self.assertEqual(split_of_video(ROOT,val),'val')
        with self.assertRaises(PermissionError): enforce_test_protection(ROOT,test,allow_test=False)
        self.assertEqual(enforce_test_protection(ROOT,test,allow_test=True),'test')

    def test_unique_gt_lookup(self):
        split=json.loads((ROOT/'reference/repeat01_config/split.json').read_text(encoding='utf-8')); key=split['val_videos'][0]
        ref,sk=load_video_skeleton(ROOT/'unified_3d_all_envs.npz',video_key=key)
        rec=load_ground_truth(ROOT/'unified_3d_all_envs.npz',video_key=key)
        self.assertEqual(rec.video_key,key); self.assertEqual(len(rec.labels),len(sk)); self.assertEqual(sk.shape[1:],(17,3)); self.assertEqual(ref.video_key,key)

    def test_public_api_skeleton(self):
        out=infer_skeleton3d(np.zeros((64,17,3),np.float32),project_root=ROOT,device='cpu')
        self.assertEqual(out['schema_version'],'2.0'); self.assertEqual(out['model_status']['dbn'],'ok'); self.assertEqual(len(out['stds']),1)

if __name__=='__main__': unittest.main()
