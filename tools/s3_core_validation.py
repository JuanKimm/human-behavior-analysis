"""Phase-3 core validation on a validation split video only (never test by default)."""
from __future__ import annotations
import argparse, json, sys, tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src'))
from runtime.ground_truth import load_video_skeleton
from runtime.test_guard import split_of_video
from dd_r01.api import run_skeleton3d_job

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--device',default='auto'); ap.add_argument('--video-key'); ap.add_argument('--output-root',default=str(ROOT/'outputs'/'s3_core_validation')); a=ap.parse_args()
    split=json.loads((ROOT/'reference/repeat01_config/split.json').read_text(encoding='utf-8')); key=a.video_key or split['val_videos'][0]
    if split_of_video(ROOT,key)!='val': raise SystemExit('s3_core_validation accepts validation videos only')
    ref,x=load_video_skeleton(ROOT/'unified_3d_all_envs.npz',video_key=key)
    pred=run_skeleton3d_job(x,ROOT,a.device,a.output_root,'predict','research',video_key=key,source='s3_predict')
    eva=run_skeleton3d_job(x,ROOT,a.device,a.output_root,'evaluate','research',ROOT/'unified_3d_all_envs.npz',key,None,None,'s3_evaluate')
    p=json.load(open(Path(pred['run_dir'])/'result.json',encoding='utf-8')); e=json.load(open(Path(eva['run_dir'])/'result.json',encoding='utf-8'))
    for model in ('tcn','stds','dbn'):
        if p[model]!=e[model]: raise SystemExit(f'GT leakage check failed: {model}')
    print('PASS S3 core validation',key); print(pred['run_dir']); print(eva['run_dir'])
if __name__=='__main__': main()
