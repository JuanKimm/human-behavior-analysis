from __future__ import annotations
from collections import Counter
import importlib.util
from pathlib import Path
import sys
import numpy as np
from .contracts import TrackedPose


def load_tracker(root):
    path = Path(root).resolve() / 'mtmct_delayed_realtime_v4-2-2.py'
    name = '_human_behavior_mtmct_legacy'
    if name in sys.modules:
        module = sys.modules[name]
        if Path(module.__file__).resolve() != path:
            raise RuntimeError('Another tracker root is already loaded')
        return module
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None); raise
    return module


class PosePredictor:
    """Capture the pose rows from the SAME Results used by the legacy bbox detector."""
    def __init__(self, model):
        self.model, self.last_poses = model, []

    def predict(self, **kwargs):
        results = list(self.model.predict(**kwargs))
        poses = []
        for r in results:
            n = len(r.boxes)
            if n == 0:
                xy, conf = np.empty((0, 17, 2), np.float32), np.empty((0, 17), np.float32)
            else:
                if r.keypoints is None or r.keypoints.conf is None:
                    raise ValueError('YOLO pose keypoints/confidences are required')
                xy = r.keypoints.xy.cpu().numpy().copy()
                conf = r.keypoints.conf.cpu().numpy().copy()
            if xy.shape != (n, 17, 2) or conf.shape != (n, 17):
                raise ValueError('Expected COCO17 pose rows aligned with detector boxes')
            if not np.isfinite(xy).all() or not np.isfinite(conf).all() or np.any((conf < 0) | (conf > 1)):
                raise ValueError('Invalid pose values')
            poses.append((xy, conf))
        self.last_poses = poses
        return results


class TrackingAdapter:
    def __init__(self, legacy, model, encoder, calibration, fps, args, event_sink=None, cfg=None):
        self.legacy, self.cfg = legacy, cfg or legacy.Config()
        self.predictor, self.encoder, self.calibration, self.args = PosePredictor(model), encoder, calibration, args
        self.trackers = {cam: legacy.make_local_tracker(fps, self.cfg) for cam in (1, 2)}
        self.guard, self.ticks = legacy.SegmentGuard(self.cfg), Counter()
        self.manager = legacy.GlobalTracker(self.cfg, event_sink)
        self.buffer, self.timings = legacy.FixedLagBuffer(self.cfg.delay_seconds, fps), legacy.Timings()
        self.pending = {}

    def push(self, packet):
        if not all(packet.fresh.values()):
            raise ValueError('Integration accepts only fresh sequential source frames')
        detections, _ = self.legacy.detect_batch(self.predictor, self.encoder, packet, self.calibration,
                                                self.trackers, self.ticks, self.args, self.cfg, self.timings)
        self.pending[packet.sequence] = dict(zip((1, 2), self.predictor.last_poses))
        for cam in (1, 2):
            rows = self.legacy.local_update(self.trackers[cam], detections[cam], packet.frames[cam])
            self.guard.update(rows, packet.timestamp)
            packet.observations[cam] = rows
            self.ticks[cam] += 1
        self.buffer.push(packet)
        while self.buffer.ready():
            yield self._convert(self.buffer.pop(self.manager))

    def flush(self):
        while self.buffer.packets:
            yield self._convert(self.buffer.pop(self.manager, flush=True))

    def _convert(self, result):
        packet, labels, lookahead, tail = result
        poses = self.pending.pop(packet.sequence)
        rows = []
        for cam, labels_cam in labels.items():
            xy, conf = poses[cam]
            for obs, gid, _ in labels_cam:
                if obs.bbox is None or gid is None or gid < 1:
                    continue
                i = obs.detector_index
                if not 0 <= i < len(xy):
                    raise RuntimeError('Lost detector-to-pose row mapping')
                rows.append(TrackedPose(packet.sequence, packet.timestamp, cam, int(gid), int(obs.local_id),
                                        int(obs.epoch), i, obs.image_size, obs.bbox, obs.confidence,
                                        xy[i], conf[i], bool(obs.partial), bool(obs.occluded)))
        return packet, rows, lookahead, tail
