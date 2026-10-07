from pathlib import Path
import sys, numpy as np
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src'))
from runtime.engine import FallRiskInferenceEngine
eng=FallRiskInferenceEngine(ROOT,device='auto')
r=eng.infer_skeleton_sequence(np.zeros((64,17,3),np.float32),source='smoke')
assert len(r.tcn)==7 and len(r.stds)==1 and len(r.dbn)==1
assert r.status=='ok' and r.model_status=={'tcn':'ok','stds':'ok','dbn':'ok'}
assert eng.infer_skeleton_sequence(np.zeros((15,17,3),np.float32)).status=='insufficient_length'
assert eng.infer_skeleton_sequence(np.zeros((16,17,3),np.float32)).status=='partial'
print('PASS',r.device,r.dbn[0].label)
