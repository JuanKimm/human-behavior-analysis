"""Run the unchanged original fall module once per input video.

There are no global-ID sequences, identity boundaries, camera rankings, or
additional detection-quality rules in this adapter.
"""
from __future__ import annotations
from pathlib import Path
import json
from runtime.evaluation import make_timeline


class OriginalVideoFallAnalyzer:
    def __init__(self, root, device='cpu'):
        from runtime.video_pipeline import VideoInferenceModule
        # Keep the original YOLO extraction, zero fill, lifting, and inference.
        # Reuse its models across complete videos; its engine resets DBN per call.
        self.module = VideoInferenceModule(Path(root), device=device)

    def analyze_video(self, video_path, camera_id, expected_frames, expected_fps):
        result, arrays = self.module.run_video(video_path)
        inference = result.to_dict()
        if inference['frame_count'] != expected_frames:
            raise RuntimeError('Original fall module decoded a different frame count than MTMCT')
        if inference['fps'] is None or abs(float(inference['fps']) - expected_fps) > 1e-6:
            raise RuntimeError('Original fall module FPS differs from MTMCT')
        # Camera ID is only the source-video label; it is never a selection score.
        timeline = []
        for row in make_timeline(inference):
            probabilities = row['dbn_probabilities']
            timeline.append(dict(camera_id=camera_id, frame_index=row['frame_idx'],
                timestamp=row['timestamp_sec'], label=row['dbn_label'],
                status=row['dbn_status'] or 'warmup', decision_frame=row['dbn_decision_frame'],
                probabilities=None if probabilities is None else json.loads(probabilities)))
        return inference, arrays, timeline
