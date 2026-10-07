from __future__ import annotations
from pathlib import Path
import json
from preprocessing.yolo_pose import YOLOPoseExtractor
from preprocessing.videopose3d import VideoPose3DLifter
from preprocessing.skeleton import validate_skeleton_sequence
from .engine import FallRiskInferenceEngine
from .env_info import resolve_device, environment_info, format_environment
from .environment_contract import verify_preprocessing_environment

class VideoInferenceModule:
    def __init__(self, project_root: str | Path, device: str | None=None, verbose: bool=True):
        self.root=Path(project_root).resolve(); cfg=json.loads((self.root/'config/module_config.json').read_text(encoding='utf-8'))
        env_contract=verify_preprocessing_environment(self.root)
        resolved,_=resolve_device(device,cfg.get('device','auto')); self.device_name=resolved; self.environment=environment_info(resolved); self.environment['preprocessing_contract']=env_contract
        if verbose: print(format_environment(self.environment))
        pose_cfg=cfg.get('pose',{}); vp=cfg.get('videopose3d',{})
        self.pose2d=YOLOPoseExtractor(self.root/cfg['paths']['yolo_weights'],device=resolved,
                                      person_selection=pose_cfg.get('person_selection','first_detection'),
                                      no_detection_fill=pose_cfg.get('no_detection_fill','zeros'))
        self.lifter=VideoPose3DLifter(self.root/cfg['paths']['videopose3d_weights'],device=resolved,
                                      filter_widths=vp.get('filter_widths',[3,3,3,3,3]),causal=vp.get('causal',False),channels=vp.get('channels',1024))
        expected_rf=vp.get('receptive_field_frames')
        if expected_rf is not None and int(expected_rf)!=self.lifter.receptive_field:
            raise ValueError(f'VideoPose3D receptive field mismatch: config={expected_rf}, actual={self.lifter.receptive_field}')
        self.engine=FallRiskInferenceEngine(self.root,device=resolved)

    def run_video(self, video_path: str | Path):
        pose=self.pose2d.extract_video(video_path); w,h=pose.resolution
        skel3d=validate_skeleton_sequence(self.lifter.lift(pose.keypoints,w,h))
        result=self.engine.infer_skeleton_sequence(skel3d,source=str(video_path),fps=pose.fps,detected=pose.detected,confidence=pose.confidence,diagnostics=pose.diagnostics)
        return result,{'detected':pose.detected,'confidence':pose.confidence,'skeletons_3d':skel3d}
