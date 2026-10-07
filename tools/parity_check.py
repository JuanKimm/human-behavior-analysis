"""Compare runtime ST-DS/DBN observations against a training-time observation CSV.
Uses unique env::video_name (or video_idx), never bare video_name.
"""
from __future__ import annotations
import argparse, csv, sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src'))
from runtime.engine import FallRiskInferenceEngine
from dbn.model_dbn import FinalGaussianDBN

FEAT=[('t4','T4'),('sp','S_P'),('sd','S_D')]
STDS=[('Normal','stds_normal_prob'),('Precursor','stds_precursor_prob'),('Danger','stds_danger_prob')]
LABELS=('Normal','Precursor','Danger')

def load_unified(path):
 d=np.load(path,allow_pickle=True)
 return {k:d[k] for k in ('skeletons','video_names','env_names','video_starts','video_lengths')}

def load_csv(path):
 with open(path,'r',encoding='utf-8-sig',newline='') as f: rows=list(csv.DictReader(f))
 if not rows: raise ValueError('CSV empty')
 if 'env_name' not in rows[0] or 'video_idx' not in rows[0]: raise KeyError('CSV must contain env_name and video_idx for unique selection')
 return rows

def pick(rows,uni,video_key=None,video_idx=None):
 if video_idx is not None:
  i=int(video_idx); key=f"{str(uni['env_names'][i])}::{str(uni['video_names'][i])}"
 elif video_key:
  if '::' not in video_key: raise ValueError('--video-key must be env::video_name')
  env,name=video_key.split('::',1); idx=np.where((uni['env_names'].astype(str)==env)&(uni['video_names'].astype(str)==name))[0]
  if len(idx)!=1: raise ValueError(f'unique key matches {len(idx)} videos')
  i=int(idx[0]); key=video_key
 else:
  r=rows[0]; i=int(r['video_idx']); key=f"{r['env_name']}::{r['video_name']}"
 return i,key

def summarize(tag,diffs,tol):
 d=np.asarray(diffs,float); over=int((d>tol).sum()); print(f"  {'OK' if over==0 else 'FAIL'} {tag:22s} max={d.max():.3e} mean={d.mean():.3e} over={over}/{len(d)}"); return over==0

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--unified',required=True); ap.add_argument('--csv',required=True); ap.add_argument('--video-key'); ap.add_argument('--video-idx',type=int); ap.add_argument('--device',default='auto'); ap.add_argument('--tol',type=float,default=1e-3); a=ap.parse_args()
 uni=load_unified(a.unified); rows=load_csv(a.csv); vi,key=pick(rows,uni,a.video_key,a.video_idx)
 s0=int(uni['video_starts'][vi]); n=int(uni['video_lengths'][vi]); sk=np.asarray(uni['skeletons'][s0:s0+n],np.float32)
 eng=FallRiskInferenceEngine(ROOT,device=a.device); out=eng.infer_skeleton_sequence(sk,source='parity::'+key).to_dict(); mod={int(w['window_start']):w for w in out['dbn']}
 vrows=sorted((r for r in rows if int(r['video_idx'])==vi and r['env_name']==key.split('::',1)[0] and r['video_name']==key.split('::',1)[1]),key=lambda r:int(r['window_abs_start']))
 pairs=[]
 for r in vrows:
  rel=int(r['window_abs_start'])-s0
  if rel in mod: pairs.append((rel,r,mod[rel]))
 if not pairs: raise SystemExit('no overlapping windows')
 dbn=FinalGaussianDBN.load(ROOT/'checkpoints/dbn/final_dbn.npz')
 off=[{'T4':float(r['T4']),'S_P':float(r['S_P']),'S_D':float(r['S_D']),'target_label':0,'video_idx':vi,'window_abs_start':int(r['window_abs_start'])} for _,r,_ in pairs]; post=dbn.filter(off)
 ok=True
 for key2,col in FEAT: ok &= summarize(col,[abs(m[key2]-float(r[col])) for _,r,m in pairs],a.tol)
 for i,(lab,col) in enumerate(STDS): ok &= summarize('STDS '+lab,[abs(m['stds_probabilities'][i]-float(r[col])) for _,r,m in pairs],a.tol)
 ok &= summarize('DBN posterior',[abs(np.asarray(m['probabilities'])-post[j]).max() for j,(_,_,m) in enumerate(pairs)],a.tol)
 agree=sum(1 for j,(_,_,m) in enumerate(pairs) if m['label']==LABELS[int(post[j].argmax())]); ok &= agree==len(pairs); print('label agreement',agree,'/',len(pairs))
 print('PASS' if ok else 'FAIL',key); raise SystemExit(0 if ok else 1)
if __name__=='__main__': main()
