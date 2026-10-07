"""Aggregate direct-3D evaluation for train/val; test is locked unless --allow-test."""
from __future__ import annotations
import argparse,csv,json,sys,time
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src'))
from runtime.engine import FallRiskInferenceEngine
from runtime.ground_truth import load_video_skeleton,load_ground_truth
from runtime.evaluation import make_timeline,compute_metrics,transition_summary
from runtime.test_guard import enforce_test_protection
from runtime.provenance import model_hashes,dataset_hashes,project_versions

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--split',choices=['train','val','test'],default='val'); ap.add_argument('--allow-test',action='store_true'); ap.add_argument('--device',default='auto'); ap.add_argument('--output-dir',required=True); ap.add_argument('--max-videos',type=int,default=None); a=ap.parse_args()
 split=json.loads((ROOT/'reference/repeat01_config/split.json').read_text(encoding='utf-8')); keys=list(map(str,split[f'{a.split}_videos'])); keys=keys[:a.max_videos] if a.max_videos else keys
 if a.split=='test' and not a.allow_test: raise PermissionError('test split locked until Gate-3')
 out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True); eng=FallRiskInferenceEngine(ROOT,device=a.device)
 all_rows=[]; trans=[]; per=[]; t0=time.perf_counter()
 for idx,key in enumerate(keys,1):
  enforce_test_protection(ROOT,key,allow_test=a.allow_test)
  ref,x=load_video_skeleton(ROOT/'unified_3d_all_envs.npz',video_key=key)
  pred=eng.infer_skeleton_sequence(x,source=key).to_dict()
  gt=load_ground_truth(ROOT/'unified_3d_all_envs.npz',video_key=key)
  tl=make_timeline(pred,gt.labels); m=compute_metrics(tl); per.append({'video_key':key,'frames':len(x),'status':pred['status'],'metrics':m})
  all_rows.extend(tl)
  for r in transition_summary(tl,None): r['video_key']=key; trans.append(r)
  print(f'{idx}/{len(keys)} {key} frames={len(x)} status={pred["status"]}')
 summary={'created_utc':datetime.now(timezone.utc).isoformat(),'split':a.split,'device':a.device,'video_count':len(keys),'runtime_seconds':time.perf_counter()-t0,'aggregate_metrics':compute_metrics(all_rows),'per_video':per,'versions':project_versions(ROOT),'models':model_hashes(ROOT),'datasets':dataset_hashes(ROOT,ROOT/'unified_3d_all_envs.npz')}
 (out/'split_metrics.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
 fields=['video_key','model','from_state','to_state','gt_transition_frame','pred_transition_frame','delay_frames','status']
 with (out/'transition_summary_all.csv').open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore'); w.writeheader(); w.writerows(trans)
 print('WROTE',out/'split_metrics.json'); print('WROTE',out/'transition_summary_all.csv')
if __name__=='__main__': main()
