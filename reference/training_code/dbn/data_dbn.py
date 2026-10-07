# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import hashlib
from pathlib import Path
from dataclasses import dataclass
from typing import Sequence
import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class WindowRecord:
    video_idx: int
    local_start: int
    abs_start: int
    abs_end: int
    target_label: int


def video_indices_from_split(split_json, data=None):
    split_json=Path(split_json)
    with open(split_json, "r", encoding="utf-8") as f:
        s = json.load(f)
    splits=(
        np.asarray(s["train_video_indices"], dtype=np.int64),
        np.asarray(s["val_video_indices"], dtype=np.int64),
        np.asarray(s["test_video_indices"], dtype=np.int64),
    )
    flat=np.concatenate(splits)
    if len(np.unique(flat)) != len(flat): raise ValueError('DBN split contains duplicate or overlapping video indices')
    if data is not None:
        if s.get('num_videos') is not None and int(s['num_videos']) != len(data['video_names']): raise ValueError('DBN split num_videos does not match unified dataset')
        if np.any(flat<0) or np.any(flat>=len(data['video_names'])): raise ValueError('DBN split index out of range')
        names=s.get('video_names')
        if names is not None and list(names)!=list(data['video_names']): raise ValueError('DBN split video_names do not match dataset order')
        envs=s.get('env_names')
        if envs is not None and list(envs)!=list(data['env_names']): raise ValueError('DBN split env_names do not match dataset order')
        if s.get('dataset_sha256') and s.get('dataset_file'):
            candidate=split_json.parent.parent.parent / s['dataset_file']
            if not candidate.exists(): candidate=split_json.parent.parent.parent.parent / s['dataset_file']
            if candidate.exists() and hashlib.sha256(candidate.read_bytes()).hexdigest()!=s['dataset_sha256']:
                raise ValueError(f"DBN split dataset SHA-256 mismatch: {candidate}")
    return splits


def build_window_records(data, video_indices: Sequence[int], window=64, stride=16):
    records = []
    labels = data["labels"]
    starts = data["video_starts"]
    lengths = data["video_lengths"]

    for vi in map(int, video_indices):
        s0 = int(starts[vi])
        length = int(lengths[vi])
        if length < window:
            continue
        for local_start in range(0, length - window + 1, stride):
            abs_start = s0 + local_start
            abs_end = abs_start + window - 1
            records.append(
                WindowRecord(
                    video_idx=vi,
                    local_start=local_start,
                    abs_start=abs_start,
                    abs_end=abs_end,
                    target_label=int(labels[abs_end]),
                )
            )
    records.sort(key=lambda r: (r.video_idx, r.local_start))
    return records


class WindowDataset(Dataset):
    def __init__(self, data, records, window=64):
        self.data, self.records, self.window = data, list(records), int(window)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        x = self.data["skeletons"][r.abs_start:r.abs_start + self.window]
        return (
            torch.from_numpy(np.asarray(x, dtype=np.float32)),
            torch.tensor(r.target_label, dtype=torch.long),
            torch.tensor(idx, dtype=torch.long),
        )
