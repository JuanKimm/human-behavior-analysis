"""Synthetic wiring smoke; never an accuracy benchmark or protected test-set evaluation."""
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from types import SimpleNamespace
import argparse
import json
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from mtmc_fall.pipeline import OfflineConfig, run_offline
from mtmc_fall.tracking import load_tracker


class ReplayPose:
    task = 'pose'
    def __init__(self, legacy):
        import cv2
        self.boxes = {}
        calibration = legacy.Calibration()
        for cam in (1, 2):
            feet = cv2.perspectiveTransform(np.array([[[3., 1.]], [[5., 1.]]]), np.linalg.inv(calibration.h[cam])).reshape(2, 2)
            self.boxes[cam] = np.array([[x - 40, y - 220, x + 40, y] for x, y in feet], np.float32)
    def __call__(self, frame, **kwargs):
        # The original extractor calls its detector on one frame at a time.
        return self.predict(source=[frame], **kwargs)

    def predict(self, source, **kwargs):
        import torch
        from ultralytics.engine.results import Results
        results = []
        for cam, frame in enumerate(source, 1):
            boxes = self.boxes[cam]
            xy = []
            for x1, y1, x2, y2 in boxes:
                xy.append(np.stack([np.linspace(x1+10, x2-10, 17), np.linspace(y1+10, y2-10, 17)], axis=1))
            kpts = np.concatenate([np.array(xy), np.full((2, 17, 1), .95 if cam == 1 else .85)], axis=2)
            detections = np.concatenate([boxes, np.full((2, 1), .95), np.zeros((2, 1))], axis=1)
            results.append(Results(orig_img=frame, path=f'synthetic-cam{cam}', names={0: 'person'},
                boxes=torch.tensor(detections, dtype=torch.float32), keypoints=torch.tensor(kpts, dtype=torch.float32)))
        return results


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--real-yolo', action='store_true'); args = parser.parse_args()
    import cv2
    legacy = load_tracker(ROOT)
    n = 2 if args.real_yolo else 84
    with TemporaryDirectory(prefix='mtmc-smoke-') as tmp:
        directory = Path(tmp); videos = tuple(directory / f'cam{c}.mp4' for c in (1, 2))
        for video in videos:
            writer = legacy.make_writer(video, 30., (1280, 720))
            try:
                for _ in range(n): writer.write(np.zeros((720, 1280, 3), np.uint8))
            finally: writer.release()
        config = OfflineConfig(ROOT, videos, ROOT / 'assets/yolo/yolo26x-pose.pt',
            ROOT / 'weights/osnet_x1_0_msmt17.pth', directory / 'outputs')
        if args.real_yolo:
            run_dir = run_offline(config)
        else:
            with patch('ultralytics.YOLO', return_value=ReplayPose(legacy)):
                run_dir = run_offline(config)
        summary = json.loads((run_dir / 'summary.json').read_text())
        assert summary['status'] == 'ok' and summary['frames'] == n, summary
        tracking = [json.loads(line) for line in (run_dir / 'tracked_poses.jsonl').read_text().splitlines()]
        assert [row['frame_index'] for row in tracking] == list(range(n))
        risks = [json.loads(line) for line in (run_dir / 'risks.jsonl').read_text().splitlines()]
        if args.real_yolo:
            assert summary['global_ids'] == [] and summary['fall_videos'] == 2
        else:
            assert len(summary['global_ids']) == 2 and summary['fall_videos'] == 2, summary
            assert summary['valid_risk_rows'] > 0
        assert len(risks) == 2 * n
        assert all('global_id' not in row and 'selected_camera_id' not in row for row in risks)
        for camera_id in (1, 2):
            with np.load(run_dir / 'fall' / f'camera{camera_id}.npz') as data:
                assert len(data['skeletons_3d']) == n
                assert len(data['detected']) == n
                assert bool(data['detected'].any()) == (not args.real_yolo)
        cap = cv2.VideoCapture(str(run_dir / 'combined_fall.mp4')); count = 0
        while cap.read()[0]: count += 1
        cap.release(); assert count == n, count
        print(json.dumps(dict(mode='actual_yolo_empty_scene' if args.real_yolo else 'replay_pose_actual_downstream_models',
            frames=n, global_ids=summary['global_ids'], fall_videos=summary['fall_videos'],
            valid_risk_rows=summary['valid_risk_rows'], output_video_frames=count, status='passed')))


if __name__ == '__main__': main()
