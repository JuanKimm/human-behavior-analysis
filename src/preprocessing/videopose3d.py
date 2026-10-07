from __future__ import annotations
from pathlib import Path
import numpy as np
import torch
from .videopose3d_model import TemporalModel

class VideoPose3DLifter:
    """VideoPose3D wrapper matching the provided training-data generation script.

    Active V1 path: COCO17 2D detector coordinates -> pretrained_h36m_detectron_coco.bin
    -> 17x3 VideoPose3D output. The supplied checkpoint is non-causal with 243-frame
    receptive field, so this file-based pipeline is not a true streaming 3D lifter.
    """
    def __init__(self, checkpoint_path: str | Path, device: str = 'cpu', filter_widths=None, causal=False, channels=1024):
        self.device = torch.device(device)
        filter_widths = list(filter_widths or [3,3,3,3,3])
        self.model = TemporalModel(
            num_joints_in=17,
            in_features=2,
            num_joints_out=17,
            filter_widths=filter_widths,
            causal=bool(causal),
            channels=int(channels),
        )
        checkpoint = torch.load(str(checkpoint_path), map_location=self.device, weights_only=True)
        state = checkpoint['model_pos'] if isinstance(checkpoint, dict) and 'model_pos' in checkpoint else checkpoint
        self.model.load_state_dict(state, strict=True)
        self.model.to(self.device).eval()
        self.receptive_field = int(self.model.receptive_field())

    @staticmethod
    def normalize_2d(keypoints: np.ndarray, width: int, height: int) -> np.ndarray:
        if width <= 0 or height <= 0: raise ValueError(f'Invalid frame size {(width,height)}')
        kp = np.asarray(keypoints, dtype=np.float32).copy()
        kp[..., 0] = kp[..., 0] / float(width) * 2.0 - 1.0
        kp[..., 1] = kp[..., 1] / float(width) * 2.0 - float(height) / float(width)
        return kp

    @torch.no_grad()
    def lift(self, keypoints_2d: np.ndarray, width: int, height: int) -> np.ndarray:
        keypoints_2d = np.asarray(keypoints_2d, dtype=np.float32)
        if keypoints_2d.ndim != 3 or keypoints_2d.shape[1:] != (17,2):
            raise ValueError(f"expected (N,17,2), got {keypoints_2d.shape}")
        if len(keypoints_2d) == 0:
            raise ValueError('영상에서 디코딩된 프레임이 없습니다 (N=0).')
        if not np.isfinite(keypoints_2d).all(): raise ValueError('2D keypoints contain NaN/Inf')
        pad = (self.receptive_field - 1) // 2
        norm = self.normalize_2d(keypoints_2d, width, height)
        padded = np.pad(norm, ((pad,pad),(0,0),(0,0)), mode='edge')
        x = torch.from_numpy(padded).unsqueeze(0).float().to(self.device)
        out = self.model(x).squeeze(0).detach().cpu().numpy().astype(np.float32)
        return out
