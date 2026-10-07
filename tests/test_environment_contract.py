import unittest, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from runtime.environment_contract import evaluate_preprocessing_contract

class EnvironmentContractTests(unittest.TestCase):
    def test_exact_match(self):
        e={'numpy':'2.2.6','opencv':'4.11.0','ultralytics':'8.4.70'}
        r=evaluate_preprocessing_contract(e,dict(e))
        self.assertEqual(r['status'],'ok'); self.assertEqual(r['mismatches'],[])
    def test_mismatch_is_detected(self):
        e={'numpy':'2.2.6','opencv':'4.11.0','ultralytics':'8.4.70'}
        c={'numpy':'2.1.3','opencv':'5.0.0','ultralytics':'8.4.165'}
        r=evaluate_preprocessing_contract(e,c)
        self.assertEqual(r['status'],'mismatch'); self.assertEqual(len(r['mismatches']),3)
if __name__=='__main__': unittest.main()
