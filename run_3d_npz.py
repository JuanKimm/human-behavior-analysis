from pathlib import Path
import argparse, sys
import numpy as np
ROOT=Path(__file__).resolve().parent; SRC=ROOT/'src'
if str(SRC) not in sys.path: sys.path.insert(0,str(SRC))
from dd_r01.api import run_skeleton3d_job
from runtime.ground_truth import load_video_skeleton

KNOWN_KEYS=('keypoints_3d','skeletons','skeleton','keypoints','pose','data','joints')

def load_npz(path):
    with np.load(path,allow_pickle=False) as d:
        key=next((k for k in KNOWN_KEYS if k in d.files),None)
        if key is None: raise KeyError(f'3D skeleton key not found. expected={KNOWN_KEYS}, actual={tuple(d.files)}')
        if any(k in d.files for k in ('video_offsets','video_names','sequence_boundaries','video_id','video_starts','video_lengths','video_idx')):
            raise ValueError('multi-video NPZ requires --video-key or --video-idx so boundaries are explicit')
        x=np.asarray(d[key],dtype=np.float32)
        if x.ndim==2 and x.shape[1]==51: x=x.reshape(-1,17,3)
        if x.ndim!=3 or x.shape[1:]!=(17,3): raise ValueError(f"'{key}' shape must be (N,17,3) or (N,51): {x.shape}")
        if not np.isfinite(x).all(): raise ValueError(f"'{key}' contains NaN/Inf")
        fps=float(np.asarray(d['fps']).item()) if 'fps' in d.files else None
        return x,fps,key

def main():
    ap=argparse.ArgumentParser(description='DD-R01 3D skeleton inference/evaluation')
    ap.add_argument('--npz',required=True)
    ap.add_argument('--mode',choices=['predict','evaluate'],default='predict')
    ap.add_argument('--profile',choices=['research','product'],default='research')
    ap.add_argument('--gt-npz',default=None)
    ap.add_argument('--video-key',default=None)
    ap.add_argument('--video-idx',type=int,default=None)
    ap.add_argument('--fps',type=float,default=None)
    ap.add_argument('--output-root',default=str(ROOT/'outputs'))
    ap.add_argument('--device',default='auto')
    ap.add_argument('--allow-test',action='store_true',help='Gate-3 approved final test only')
    a=ap.parse_args(); p=Path(a.npz)
    multi=False
    try:
        x,fps,key=load_npz(p)
    except ValueError as e:
        if 'multi-video NPZ' not in str(e) or (a.video_key is None and a.video_idx is None): raise
        ref,x=load_video_skeleton(p,video_key=a.video_key,video_idx=a.video_idx)
        fps=a.fps; key='skeletons'; multi=True
        if a.video_key is None: a.video_key=ref.video_key
    fps=a.fps if a.fps is not None else fps
    gt_npz=a.gt_npz or (str(p) if multi and a.mode=='evaluate' else None)
    info=run_skeleton3d_job(x,ROOT,a.device,a.output_root,a.mode,a.profile,gt_npz,a.video_key,a.video_idx,fps,source=str(p),input_path=str(p),allow_test=a.allow_test)
    st=info['validation_status']
    if st['status']!='ok': raise SystemExit(f'RUN FAILED: {st}')
    print(f"[완료] input_key={key} / frames={len(x)} / run_dir={info['run_dir']} / inference_status={st['inference_status']}")

if __name__=='__main__': main()
