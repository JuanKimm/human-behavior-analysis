"""Reconstruct TCN 16F/stride4 clips from unified_3d_all_envs.npz.
This is a provenance/reproducibility tool. It does not claim byte identity with the
original intermediate NPZ files, which are not bundled. It reproduces clip boundaries,
last-frame labels, video keys and split counts recorded in repeat01 config.
"""
from pathlib import Path
import argparse, json
import numpy as np
ROOT=Path(__file__).resolve().parents[1]

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--unified',default=str(ROOT/'unified_3d_all_envs.npz')); ap.add_argument('--output-dir'); ap.add_argument('--count-only',action='store_true'); a=ap.parse_args()
    if not a.count_only and not a.output_dir: ap.error('--output-dir is required unless --count-only is used')
    out=Path(a.output_dir) if a.output_dir else None
    if out: out.mkdir(parents=True,exist_ok=True)
    with np.load(a.unified,allow_pickle=True) as d:
        sk=d['skeletons'].astype(np.float32); lab=d['labels'].astype(np.int64); names=d['video_names'].astype(str); envs=d['env_names'].astype(str); starts=d['video_starts'].astype(np.int64); lens=d['video_lengths'].astype(np.int64)
        groups={}
        for env in np.unique(envs): groups[str(env)]=[]
        for i,(env,name,s,n) in enumerate(zip(envs,names,starts,lens)):
            for st in range(0,int(n)-16+1,4):
                abs_s=int(s)+st; groups[str(env)].append((sk[abs_s:abs_s+16],int(lab[abs_s+15]),str(name),st))
    for env,rows in groups.items():
        print(env,len(rows))
        if not a.count_only:
            X=np.stack([r[0] for r in rows]).astype(np.float32); y=np.asarray([r[1] for r in rows],np.int64); vn=np.asarray([r[2] for r in rows]); sf=np.asarray([r[3] for r in rows],np.int64)
            np.savez_compressed(out/f'fall_dataset_3D_win16_stride4_with_conf_{env}.npz',X=X,y=y,video_name=vn,start_frame=sf,env_name=np.asarray([env]))
    split=json.loads((ROOT/'reference/repeat01_config/split.json').read_text(encoding='utf-8')); cfg=json.loads((ROOT/'reference/repeat01_config/01_tcn/config.json').read_text(encoding='utf-8'))
    keys=np.asarray([f'{e}::{n}' for e,n in zip(envs,names)]); counts={k:0 for k in ('train','val','test')}
    sets={k:set(split[f'{k}_videos']) for k in counts}
    for key,n in zip(keys,lens):
        c=max(0,(int(n)-16)//4+1)
        for k,s in sets.items():
            if key in s: counts[k]+=c; break
    expected={'train':cfg['train_clips'],'val':cfg['val_clips'],'test':cfg['reserved_test_clips']}
    print('counts',counts,'expected',expected)
    if counts!=expected: raise SystemExit('clip count mismatch')
if __name__=='__main__': main()
