# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import hashlib
from pathlib import Path
import numpy as np

NUM_CLASSES = 3
CLASS_NAMES = ["정상", "전조", "위험"]
PRECURSOR_LABEL = 1


def load_unified(path):
    with np.load(Path(path), allow_pickle=False) as d:
        required={"skeletons","labels","video_idx","video_names","env_names","video_starts","video_lengths"}
        missing=required.difference(d.files)
        if missing: raise ValueError(f"Unified NPZ missing keys: {sorted(missing)}")
        out={"skeletons":d["skeletons"].astype(np.float32),"labels":d["labels"].astype(np.int64),
             "video_idx":d["video_idx"].astype(np.int64),"video_names":d["video_names"].astype(str),
             "env_names":d["env_names"].astype(str),"video_starts":d["video_starts"].astype(np.int64),
             "video_lengths":d["video_lengths"].astype(np.int64)}
    n=len(out['skeletons']); nv=len(out['video_names'])
    if out['skeletons'].shape[1:]!=(17,3) or not np.isfinite(out['skeletons']).all(): raise ValueError('Unified skeletons must be finite (N,17,3)')
    if len(out['labels'])!=n or len(out['video_idx'])!=n or np.any((out['labels']<0)|(out['labels']>=NUM_CLASSES)): raise ValueError('Unified labels/video_idx length or class range invalid')
    if len(out['video_starts'])!=nv or len(out['video_lengths'])!=nv or len(out['env_names'])!=nv: raise ValueError('Unified per-video metadata lengths mismatch')
    video_keys=list(zip(out['env_names'].tolist(),out['video_names'].tolist()))
    if len(set(video_keys))!=nv: raise ValueError('Unified dataset contains duplicate (environment, video) identifiers')
    if np.any(out['video_starts']<0) or np.any(out['video_lengths']<0) or np.any(out['video_starts']+out['video_lengths']>n): raise ValueError('Unified video range out of bounds')
    if np.any((out['video_idx']<0)|(out['video_idx']>=nv)): raise ValueError('Unified video_idx out of bounds')
    for i,(start,length) in enumerate(zip(out['video_starts'],out['video_lengths'])):
        if not np.all(out['video_idx'][start:start+length]==i): raise ValueError(f'video_idx does not match range metadata for video {i}')
    return out


def split_videos_from_json(data, split_json):
    split_json=Path(split_json)
    with open(split_json, "r", encoding="utf-8") as f:
        split = json.load(f)

    tr = np.asarray(split["train_video_indices"], dtype=np.int64)
    va = np.asarray(split["val_video_indices"], dtype=np.int64)
    te = np.asarray(split["test_video_indices"], dtype=np.int64)

    all_ids = set(range(len(data["video_names"])))
    for name, ids in [("train", tr), ("val", va), ("test", te)]:
        if len(np.unique(ids)) != len(ids): raise RuntimeError(f"{name} split contains duplicate video indices")
        if not set(ids.tolist()).issubset(all_ids):
            raise RuntimeError(f"{name} split video index 범위 오류")

    if set(tr) & set(va) or set(tr) & set(te) or set(va) & set(te):
        raise RuntimeError("ST-DS split video leakage")
    if split.get('num_videos') is not None and int(split['num_videos']) != len(data['video_names']):
        raise RuntimeError(f"split num_videos={split['num_videos']} but dataset has {len(data['video_names'])}")
    listed=split.get('video_names')
    if listed is not None and list(listed)!=list(data['video_names']): raise RuntimeError('split video_names order/content does not match dataset')
    listed_envs=split.get('env_names')
    if listed_envs is not None and list(listed_envs)!=list(data['env_names']): raise RuntimeError('split env_names order/content does not match dataset')
    if split.get('dataset_sha256') and split.get('dataset_file'):
        dataset_path=split_json.parent.parent.parent / split['dataset_file']
        if not dataset_path.exists(): dataset_path=split_json.parent.parent.parent.parent / split['dataset_file']
        if dataset_path.exists() and hashlib.sha256(dataset_path.read_bytes()).hexdigest()!=split['dataset_sha256']:
            raise RuntimeError(f'split dataset SHA-256 mismatch: {dataset_path}')
    return tr, va, te, split


def _uniform_starts(length, window, stride):
    if length < window:
        return []
    return list(range(0, length - window + 1, stride))


def _mixed_starts(labels, window, stride_coarse, stride_dense):
    length = len(labels)
    if length < window:
        return []
    starts = set(range(0, length - window + 1, stride_coarse))
    for s in range(0, length - window + 1, stride_dense):
        if np.any(labels[s:s + window] == PRECURSOR_LABEL):
            starts.add(s)
    return sorted(starts)


def build_sequence_clips(
    data, video_ids, window=64, stride=8, mixed=False, stride_dense=2
):
    skeletons = data["skeletons"]
    labels = data["labels"]
    starts_arr = data["video_starts"]
    lengths_arr = data["video_lengths"]

    all_x, all_y, all_vid, all_fid = [], [], [], []

    for vi in map(int, video_ids):
        s0 = int(starts_arr[vi])
        length = int(lengths_arr[vi])
        if length < window:
            continue

        skel = skeletons[s0:s0 + length]
        lab = labels[s0:s0 + length]
        starts = (
            _mixed_starts(lab, window, stride, stride_dense)
            if mixed else _uniform_starts(length, window, stride)
        )

        for s in starts:
            e = s + window
            all_x.append(skel[s:e])
            all_y.append(lab[s:e])
            all_vid.append(vi)
            all_fid.append(np.arange(s0 + s, s0 + e, dtype=np.int64))

    if not all_x:
        c = int(skeletons.shape[-1])
        return (
            np.zeros((0, window, 17, c), dtype=np.float32),
            np.zeros((0, window), dtype=np.int64),
            np.zeros((0,), dtype=np.int64),
            np.zeros((0, window), dtype=np.int64),
        )

    return (
        np.stack(all_x).astype(np.float32),
        np.stack(all_y).astype(np.int64),
        np.asarray(all_vid, dtype=np.int64),
        np.stack(all_fid).astype(np.int64),
    )


def precursor_segment_lengths(data, video_ids):
    labels = data["labels"]
    starts = data["video_starts"]
    lengths = data["video_lengths"]
    out = []
    for vi in map(int, video_ids):
        s0, length = int(starts[vi]), int(lengths[vi])
        y = labels[s0:s0 + length]
        run = 0
        for value in y:
            if int(value) == PRECURSOR_LABEL:
                run += 1
            elif run:
                out.append(run)
                run = 0
        if run:
            out.append(run)
    return np.asarray(out, dtype=np.int64)


def describe_segment_lengths(lengths):
    if len(lengths) == 0:
        return {"count": 0}
    return {
        "count": int(len(lengths)),
        "min": int(np.min(lengths)),
        "median": float(np.median(lengths)),
        "mean": float(np.mean(lengths)),
        "max": int(np.max(lengths)),
    }
