from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import cv2
import numpy as np

@dataclass
class Pose2DResult:
    keypoints: np.ndarray
    confidence: np.ndarray
    detected: np.ndarray
    fps: float
    resolution: tuple[int, int]
    diagnostics: list[str]

class YOLOPoseExtractor:
    """Faithful V1 wrapper of the user's original YOLO26x-pose extraction logic.

    - Uses the first detected person only.
    - Missing detections remain all-zero keypoints/confidence for that frame.
    - Produces COCO17 pixel coordinates.
    """
    def __init__(self, weights: str | Path, device: str | None = None, person_selection: str = 'first_detection', no_detection_fill: str = 'zeros'):
        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise ImportError("ultralytics가 필요합니다. `pip install ultralytics`를 실행하세요.") from e
        if person_selection != 'first_detection': raise ValueError(f'Unsupported person_selection: {person_selection}')
        if no_detection_fill != 'zeros': raise ValueError(f'Unsupported no_detection_fill: {no_detection_fill}')
        self.weights = str(weights)
        self.model = YOLO(self.weights, verbose=False)
        self.device = device
        self.person_selection = person_selection
        self.no_detection_fill = no_detection_fill

    def extract_video(self, video_path: str | Path) -> Pose2DResult:
        video_path = str(video_path)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"비디오를 열 수 없습니다: {video_path}")

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        raw_width = float(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); raw_height = float(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width = int(raw_width); height = int(raw_height)
        if not np.isfinite(fps) or fps <= 0 or not np.isfinite(raw_width) or not np.isfinite(raw_height) or width <= 0 or height <= 0:
            cap.release()
            raise ValueError(f'잘못된 영상 메타데이터: fps={fps}, resolution={(width,height)}')

        # Some codecs report frame_count=0. Fall back to dynamic lists in that case.
        if total > 0:
            keypoints = np.zeros((total, 17, 2), dtype=np.float32)
            confidence = np.zeros((total, 17), dtype=np.float32)
            detected = np.zeros((total,), dtype=bool)
            dynamic = False
        else:
            keypoints, confidence, detected = [], [], []
            dynamic = True

        i = 0
        try:
          while True:
            ok, frame = cap.read()
            if not ok: break
            kwargs = {"verbose": False}
            if self.device: kwargs["device"] = self.device
            results = self.model(frame, **kwargs)
            xy = np.zeros((17,2), dtype=np.float32)
            cf = np.zeros((17,), dtype=np.float32)
            det = False
            if results and results[0].keypoints is not None and results[0].keypoints.xy.shape[0] > 0:
                kp = results[0].keypoints
                xy = kp.xy[0].detach().cpu().numpy().astype(np.float32)
                if kp.conf is not None:
                    cf = kp.conf[0].detach().cpu().numpy().astype(np.float32)
                det = True

            if dynamic:
                keypoints.append(xy); confidence.append(cf); detected.append(det)
            else:
                if i >= len(keypoints):
                    # Defensive fallback if container metadata under-reported frame count.
                    keypoints = list(keypoints); confidence = list(confidence); detected = list(detected)
                    dynamic = True
                    keypoints.append(xy); confidence.append(cf); detected.append(det)
                else:
                    keypoints[i] = xy; confidence[i] = cf; detected[i] = det
            i += 1
        finally:
            cap.release()
        if i == 0:
            raise ValueError(f'영상에서 디코딩된 프레임이 없습니다: {video_path}')
        if dynamic:
            keypoints = np.asarray(keypoints, dtype=np.float32)[:i]
            confidence = np.asarray(confidence, dtype=np.float32)[:i]
            detected = np.asarray(detected, dtype=bool)[:i]
        else:
            keypoints = keypoints[:i]
            confidence = confidence[:i]
            detected = detected[:i]

        diagnostics=[]
        if total > 0 and i != total:
            diagnostics.append(f'CAP_PROP_FRAME_COUNT={total}, decoded_frames={i}; EOF/decode failure cause cannot be distinguished by this backend')
        return Pose2DResult(
            keypoints=keypoints,
            confidence=confidence,
            detected=detected,
            fps=fps,
            resolution=(width, height),
            diagnostics=diagnostics,
        )
