"""Dependency-light checks for Phase-1 input and window contract."""
import importlib.util
import tempfile
import unittest
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))
from run_3d_npz import load_npz

class PhaseOneContracts(unittest.TestCase):
    def test_window_counts(self):
        expected = {0:(0,0),15:(0,0),16:(1,0),63:(6,0),64:(7,1),65:(7,1),80:(9,2),448:(55,25)}
        for n, want in expected.items():
            got = (max(0,(n-16)//8+1), max(0,(n-64)//16+1))
            self.assertEqual(got,want,n)

    def test_npz_empty_vs_empty_skeleton(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'input.npz'
            np.savez(p, skeletons=np.zeros((0,17,3),np.float32))
            x, fps, key = load_npz(p)
            self.assertEqual(x.shape,(0,17,3)); self.assertIsNone(fps); self.assertEqual(key,'skeletons')
            np.savez(p, unknown=np.zeros((1,)))
            with self.assertRaises(KeyError): load_npz(p)

    def test_npz_rejects_malformed_values_and_boundaries(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'input.npz'
            np.savez(p, skeletons=np.full((1,17,3),np.nan))
            with self.assertRaises(ValueError): load_npz(p)
            np.savez(p, skeletons=np.zeros((1,17,3)),video_starts=np.array([0]),video_lengths=np.array([1]))
            with self.assertRaises(ValueError): load_npz(p)

    def test_annotation_intervals_do_not_create_phantom_zero(self):
        names=('build_unified_3d.py','build_fall_dataset_win64_stride8.py','build_fall_dataset_3d_stride8_win16.py')
        for name in names:
            path=ROOT/'reference'/'preprocessing_original'/name
            spec=importlib.util.spec_from_file_location('builder_'+name.replace('.','_'),path)
            mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
            self.assertEqual(mod.get_sets_for_row(['video','','','',''],1),[])
            self.assertEqual(mod.get_sets_for_row(['video','1','2','',''],1),[(1,2,None,None)])
            with self.assertRaises(ValueError): mod.get_sets_for_row(['video','1','','',''],1)
            labels=mod.build_frame_labels(5,[(1,3,None,None),(None,None,2,4)])
            self.assertEqual(labels.tolist(),[0,1,2,2,2])

if __name__ == '__main__': unittest.main()
