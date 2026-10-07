# -*- coding: utf-8 -*-
"""
Skeleton-TCN win16 데이터 로더.
split.json의 video key를 사용해 train/val/test를 정확히 재현한다.

실제 전처리 파일명 예:
  fall_dataset_3D_win16_stride4_with_conf_3AI_Office_01.npz
  fall_dataset_3D_win16_stride4_with_conf_Coffee_room_01.npz

unified_3d_all_envs.npz의 env_names와 맞추기 위해 파일명 prefix를 제거해서
환경명을 3AI_Office_01, Coffee_room_01, ... 형태로 복원한다.
"""

from __future__ import annotations

import glob
import json
import os
from pathlib import Path
import numpy as np

NUM_CLASSES = 2
CLASS_NAMES = ["정상", "낙상"]

KNOWN_PREFIXES = (
    "fall_dataset_3D_win16_stride4_with_conf_",
    "fall_dataset_3d_win16_stride4_with_conf_",
)


def to_binary_label(y):
    return (np.asarray(y) == 2).astype(np.int64)


def _infer_env_name(fp, d):
    # NPZ 안에 환경명이 있으면 그것을 최우선 사용
    for key in ("env_name", "environment", "env"):
        if key in d.files:
            arr = np.asarray(d[key]).astype(str).reshape(-1)
            uniq = np.unique(arr)
            if len(uniq) == 1:
                return str(uniq[0])

    stem = Path(fp).stem
    for prefix in KNOWN_PREFIXES:
        if stem.startswith(prefix):
            return stem[len(prefix):]

    # 이미 3AI_Office_01.npz 같은 이름이면 그대로 사용
    return stem


def load_all_npz(data_dir, max_channels=3):
    files = sorted(glob.glob(os.path.join(str(data_dir), "*.npz")))
    if not files:
        raise FileNotFoundError(f"npz 없음: {data_dir}")

    xs, ys, vids, envs = [], [], [], []

    for fp in files:
        if not any(tag in Path(fp).name.lower() for tag in ('win16_stride4','win16_stride_4')):
            raise ValueError(f"{fp}: TCN training clips must be identified as window16/stride4")
        d = np.load(fp, allow_pickle=False)
        env = _infer_env_name(fp, d)

        if "X" not in d.files or "y" not in d.files:
            raise KeyError(f"{fp}: X/y key가 없습니다. keys={d.files}")

        x = d["X"].astype(np.float32)
        y = d["y"].astype(np.int64)

        if x.ndim != 4 or x.shape[1] != 16 or x.shape[2] != 17:
            raise ValueError(
                f"{fp}: expected (N,16,17,C), actual={x.shape}"
            )
        if not np.isfinite(x).all(): raise ValueError(f"{fp}: X contains NaN/Inf")
        if not set(np.unique(y).tolist()).issubset({0,1,2}): raise ValueError(f"{fp}: unexpected labels")

        if max_channels is not None and x.shape[-1] > max_channels:
            x = x[..., :max_channels]

        if "video_name" not in d.files:
            raise KeyError(
                f"{fp}: 반복 70/15/15 split에는 video_name key가 반드시 필요합니다."
            )

        vn = d["video_name"].astype(str)

        if len(x) != len(y) or len(vn) != len(y):
            raise ValueError(
                f"{fp}: 길이 불일치 X={len(x)}, y={len(y)}, video_name={len(vn)}"
            )
        if "start_frame" in d.files:
            starts=np.asarray(d["start_frame"],dtype=np.int64).reshape(-1)
            if len(starts)!=len(vn): raise ValueError(f"{fp}: start_frame length mismatch")
            for video in np.unique(vn):
                offsets=np.sort(np.unique(starts[vn==video]))
                if len(offsets)>1 and np.any(np.diff(offsets)%4!=0):
                    raise ValueError(f"{fp}: start_frame offsets are inconsistent with stride4 for {video}")

        video_keys = np.asarray(
            [f"{env}::{v}" for v in vn],
            dtype=str,
        )

        xs.append(x)
        ys.append(y)
        vids.append(video_keys)
        envs.append(np.asarray([env] * len(y), dtype=str))

        print(
            f"  {Path(fp).name} -> env={env} | "
            f"clips={len(y):,}"
        )

    return (
        np.concatenate(xs, axis=0),
        np.concatenate(ys, axis=0),
        np.concatenate(vids, axis=0),
        np.concatenate(envs, axis=0),
    )


def indices_from_split(vid, split_json):
    with open(split_json, "r", encoding="utf-8") as f:
        split = json.load(f)

    vid = np.asarray(vid).astype(str)
    train_set = set(map(str, split["train_videos"]))
    val_set = set(map(str, split["val_videos"]))
    test_set = set(map(str, split["test_videos"]))

    available = set(vid.tolist())
    missing = (train_set | val_set | test_set) - available
    if missing:
        sample = sorted(missing)[:10]
        raise RuntimeError(
            "processed_3d_win16_stride4의 video_name/env 매칭이 "
            "unified split과 맞지 않습니다. "
            f"missing 예시={sample}"
        )

    tr = np.where(np.isin(vid, list(train_set)))[0]
    va = np.where(np.isin(vid, list(val_set)))[0]
    te = np.where(np.isin(vid, list(test_set)))[0]

    trv, vav, tev = set(vid[tr]), set(vid[va]), set(vid[te])
    if trv & vav or trv & tev or vav & tev:
        raise RuntimeError("TCN split video leakage 감지")

    return tr, va, te, split


def prep_xy_for_tcn(x):
    return np.transpose(x, (0, 3, 1, 2)).copy()
