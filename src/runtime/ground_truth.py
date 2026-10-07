from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np

LABEL_NAMES=('Normal','Precursor','Danger')

@dataclass
class VideoRecordRef:
    video_key: str
    video_idx: int
    env_name: str
    video_name: str
    start: int
    length: int

@dataclass
class GroundTruthRecord:
    video_key: str
    video_idx: int
    env_name: str
    video_name: str
    labels: np.ndarray


def resolve_video_ref(npz_path: str | Path, video_key: str | None=None, video_idx: int | None=None) -> VideoRecordRef:
    with np.load(npz_path,allow_pickle=True) as d:
        required=('video_names','video_starts','video_lengths')
        missing=[k for k in required if k not in d.files]
        if missing: raise KeyError(f'NPZ missing metadata keys: {missing}')
        names=d['video_names'].astype(str); envs=d['env_names'].astype(str) if 'env_names' in d.files else np.asarray(['']*len(names))
        starts=d['video_starts'].astype(np.int64); lengths=d['video_lengths'].astype(np.int64)
        if video_idx is None:
            if not video_key or '::' not in video_key: raise ValueError('unique video_key="env::video_name" or video_idx required')
            env,name=video_key.split('::',1); idx=np.where((envs==env)&(names==name))[0]
            if len(idx)!=1: raise KeyError(f'video_key must resolve uniquely: {video_key!r}, matches={len(idx)}')
            video_idx=int(idx[0])
        i=int(video_idx)
        if not 0<=i<len(names): raise IndexError(f'video_idx out of range: {i}')
        key=f'{envs[i]}::{names[i]}' if envs[i] else str(names[i])
        return VideoRecordRef(key,i,str(envs[i]),str(names[i]),int(starts[i]),int(lengths[i]))


def load_video_skeleton(npz_path: str | Path, video_key: str | None=None, video_idx: int | None=None):
    ref=resolve_video_ref(npz_path,video_key,video_idx)
    with np.load(npz_path,allow_pickle=True) as d:
        if 'skeletons' not in d.files: raise KeyError('NPZ has no skeletons')
        x=np.asarray(d['skeletons'][ref.start:ref.start+ref.length],dtype=np.float32)
    if x.shape!=(ref.length,17,3) or not np.isfinite(x).all(): raise ValueError(f'invalid skeleton slice: {x.shape}')
    return ref,x


def load_ground_truth(npz_path: str | Path, video_key: str | None=None, video_idx: int | None=None) -> GroundTruthRecord:
    ref=resolve_video_ref(npz_path,video_key,video_idx)
    with np.load(npz_path,allow_pickle=True) as d:
        if 'labels' not in d.files: raise KeyError('GT NPZ has no labels')
        labels=np.asarray(d['labels'][ref.start:ref.start+ref.length],dtype=np.int64)
    if len(labels)!=ref.length or not set(np.unique(labels).tolist()).issubset({0,1,2}): raise ValueError(f'invalid GT labels for {ref.video_key}')
    return GroundTruthRecord(ref.video_key,ref.video_idx,ref.env_name,ref.video_name,labels)
