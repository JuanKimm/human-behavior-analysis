# -*- coding: utf-8 -*-
from pathlib import Path
import argparse
from .train_hybrid_core import train_stds


def run_stds_stage(project_root, repeat_dir, split_json, seed, patience=15):
    project_root = Path(project_root).resolve()
    repeat_dir = Path(repeat_dir).resolve()
    unified_path = project_root / "data" / "unified_3d_all_envs.npz"
    if not unified_path.exists(): unified_path = project_root / "unified_3d_all_envs.npz"
    return train_stds(
        unified_path=unified_path,
        tcn_ckpt_path=repeat_dir / "01_tcn" / "best.pt",
        split_json=split_json,
        run_dir=repeat_dir / "02_stds",
        seed=seed,
        patience=patience,
    )

def main():
    parser=argparse.ArgumentParser(description='Train the STDS stage')
    parser.add_argument('project_root'); parser.add_argument('repeat_dir'); parser.add_argument('split_json')
    parser.add_argument('--seed',type=int,default=42); parser.add_argument('--patience',type=int,default=15)
    args=parser.parse_args()
    run_stds_stage(args.project_root,args.repeat_dir,args.split_json,args.seed,args.patience)

if __name__ == '__main__': main()
