from pathlib import Path
import argparse, sys
ROOT=Path(__file__).resolve().parent; SRC=ROOT/'src'
if str(SRC) not in sys.path: sys.path.insert(0,str(SRC))
from dd_r01.api import run_video_job


def main():
    ap=argparse.ArgumentParser(description='DD-R01 raw video batch inference/evaluation')
    ap.add_argument('--video',required=True)
    ap.add_argument('--mode',choices=['predict','evaluate'],default='predict')
    ap.add_argument('--profile',choices=['research','product'],default='research')
    ap.add_argument('--gt-npz',default=None)
    ap.add_argument('--video-key',default=None,help='evaluate: unique env::video_name')
    ap.add_argument('--video-idx',type=int,default=None)
    ap.add_argument('--output-root',default=str(ROOT/'outputs'))
    ap.add_argument('--device',default='auto')
    ap.add_argument('--allow-test',action='store_true',help='Gate-3 approved final test only')
    a=ap.parse_args()
    info=run_video_job(a.video,ROOT,a.device,a.output_root,a.mode,a.profile,a.gt_npz,a.video_key,a.video_idx,a.allow_test)
    st=info['validation_status']
    if st['status']!='ok': raise SystemExit(f"RUN FAILED: {st}")
    print(f"[완료] run_dir={info['run_dir']} / inference_status={st['inference_status']}")

if __name__=='__main__': main()
