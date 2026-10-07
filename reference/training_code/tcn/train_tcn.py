# -*- coding: utf-8 -*-
from pathlib import Path
from .train_core import train_tcn


def run_tcn_stage(project_root, repeat_dir, split_json, seed):
    project_root = Path(project_root).resolve()
    repeat_dir = Path(repeat_dir).resolve()
    return train_tcn(
        data_dir=project_root / "data" / "processed_3d_win16_stride4",
        split_json=split_json,
        run_dir=repeat_dir / "01_tcn",
        seed=seed,
        max_channels=3,
    )
