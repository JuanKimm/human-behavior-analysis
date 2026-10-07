from __future__ import annotations
from pathlib import Path
import hashlib, time
import numpy as np
from runtime.video_pipeline import VideoInferenceModule
from runtime.engine import FallRiskInferenceEngine
from runtime.ground_truth import load_ground_truth, resolve_video_ref
from runtime.bundle_writer import write_run_bundle
from runtime.test_guard import enforce_test_protection


def _root(project_root=None):
    return Path(project_root).resolve() if project_root else Path(__file__).resolve().parents[2]


def _predict_key(path):
    p=Path(path); h=hashlib.sha256(p.read_bytes()).hexdigest()[:8]
    return f'{p.stem}__{h}'


def infer_video(video_path, project_root=None, device='auto'):
    root=_root(project_root); result,_=VideoInferenceModule(root,device=device).run_video(video_path)
    return result.to_dict()


def infer_skeleton3d(skeletons, project_root=None, device='auto', source='skeleton3d', fps=None):
    root=_root(project_root); eng=FallRiskInferenceEngine(root,device=device)
    return eng.infer_skeleton_sequence(np.asarray(skeletons,dtype=np.float32),source=source,fps=fps).to_dict()


def run_video_job(video_path, project_root=None, device='auto', output_root=None, mode='predict', profile='research', gt_npz=None, video_key=None, video_idx=None, allow_test=False):
    root=_root(project_root); p=Path(video_path); gt=None
    if mode=='evaluate':
        if not gt_npz: gt_npz=root/'unified_3d_all_envs.npz'
        if video_key is None:
            video_key=resolve_video_ref(gt_npz,video_idx=video_idx).video_key
        enforce_test_protection(root,video_key,allow_test=allow_test)
    elif mode!='predict': raise ValueError("mode must be 'predict' or 'evaluate'")
    t0=time.perf_counter(); result,_=VideoInferenceModule(root,device=device).run_video(p); runtime_seconds=time.perf_counter()-t0; d=result.to_dict()
    if mode=='evaluate':
        gt=load_ground_truth(gt_npz,video_key=video_key,video_idx=video_idx)
        video_key=gt.video_key
    if not video_key: video_key=_predict_key(p)
    return write_run_bundle(root,d,p,output_root or root/'outputs',video_key,mode,profile,gt,gt_npz,create_overlay=True,runtime_info={'inference_seconds':runtime_seconds,'effective_fps':(d['frame_count']/runtime_seconds if runtime_seconds>0 else None)})


def run_skeleton3d_job(skeletons, project_root=None, device='auto', output_root=None, mode='predict', profile='research', gt_npz=None, video_key=None, video_idx=None, fps=None, source='skeleton3d', input_path=None, allow_test=False):
    root=_root(project_root); gt=None
    if mode=='evaluate':
        if not gt_npz: gt_npz=root/'unified_3d_all_envs.npz'
        if video_key is None:
            video_key=resolve_video_ref(gt_npz,video_idx=video_idx).video_key
        enforce_test_protection(root,video_key,allow_test=allow_test)
    elif mode!='predict': raise ValueError("mode must be 'predict' or 'evaluate'")
    eng=FallRiskInferenceEngine(root,device=device); t0=time.perf_counter(); d=eng.infer_skeleton_sequence(np.asarray(skeletons,dtype=np.float32),source=source,fps=fps).to_dict(); runtime_seconds=time.perf_counter()-t0
    if mode=='evaluate':
        gt=load_ground_truth(gt_npz,video_key=video_key,video_idx=video_idx); video_key=gt.video_key
    if not video_key: video_key='skeleton3d'
    return write_run_bundle(root,d,input_path,output_root or root/'outputs',video_key,mode,profile,gt,gt_npz,create_overlay=False,runtime_info={'inference_seconds':runtime_seconds,'effective_fps':(d['frame_count']/runtime_seconds if runtime_seconds>0 else None)})
