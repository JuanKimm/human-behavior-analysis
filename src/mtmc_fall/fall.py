from __future__ import annotations
from collections import defaultdict
import json
import numpy as np
from runtime.evaluation import make_timeline


class OfflineFallAnalyzer:
    """One shared lifter/engine, serial whole-sequence calls; the engine resets DBN per call."""
    def __init__(self, lifter, engine):
        self.lifter, self.engine = lifter, engine

    def analyze(self, sequence):
        poses = sequence.poses; first = poses[0]
        xy = np.stack([p.keypoints for p in poses])
        confidence = np.stack([p.confidence for p in poses])
        threshold = float(self.engine.config.get('pose_quality', {}).get('confidence_threshold', .25))
        detected = (confidence >= threshold).any(axis=1)
        skeleton = np.asarray(self.lifter.lift(xy, *first.image_size), dtype=np.float32)
        if skeleton.shape != (len(poses), 17, 3) or not np.isfinite(skeleton).all():
            raise ValueError('Lifter output does not match the input sequence')
        result = self.engine.infer_skeleton_sequence(skeleton, source=sequence.name, fps=sequence.fps,
                                                     detected=detected, confidence=confidence).to_dict()
        if int(result['frame_count']) != len(poses):
            raise RuntimeError('Inference frame count mismatch')
        windows = {r['window_end']: r for r in result['dbn']}
        timeline = []
        for row in make_timeline(result):
            decision = row['dbn_decision_frame']
            window = windows.get(decision)
            quality = None
            if window is not None:
                c = confidence[window['window_start']:window['window_end'] + 1]
                quality = {'valid_joint_ratio': float((c >= threshold).mean()),
                           'mean_joint_confidence': float(c.mean()), 'confidence_threshold': threshold}
            probability = row['dbn_probabilities']
            timeline.append(dict(frame_index=first.frame_index + row['frame_idx'],
                timestamp=(first.frame_index + row['frame_idx']) / sequence.fps,
                global_id=first.global_id, camera_id=first.camera_id, sequence_id=sequence.name,
                label=row['dbn_label'], status=row['dbn_status'] or 'warmup',
                probabilities=None if probability is None else json.loads(probability),
                decision_frame=None if decision is None else first.frame_index + decision,
                window_start=None if window is None else first.frame_index + window['window_start'],
                window_end=None if window is None else first.frame_index + window['window_end'],
                pose_quality=quality))
        return skeleton, result, timeline


def select_camera_results(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row['frame_index'], row['global_id']].append(row)
    selected = []
    for (frame, gid), cameras in sorted(groups.items()):
        if len({r['camera_id'] for r in cameras}) != len(cameras):
            raise ValueError('Duplicate per-camera decision')
        eligible = [r for r in cameras if r['status'] == 'ok' and r['label'] in ('Normal', 'Precursor', 'Danger')
                    and r['pose_quality'] is not None]
        best = max(eligible, key=lambda r: (r['pose_quality']['valid_joint_ratio'],
                    r['pose_quality']['mean_joint_confidence'], -r['camera_id']), default=None)
        selected.append(dict(frame_index=frame, timestamp=cameras[0]['timestamp'], global_id=gid,
                             selected_camera_id=None if best is None else best['camera_id'],
                             status='unavailable' if best is None else 'ok',
                             label=None if best is None else best['label'],
                             probabilities=None if best is None else best['probabilities'],
                             decision_frame=None if best is None else best['decision_frame'],
                             cameras=sorted(cameras, key=lambda r: r['camera_id'])))
    return selected
