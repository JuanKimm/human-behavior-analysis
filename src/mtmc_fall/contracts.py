from __future__ import annotations
from dataclasses import dataclass, fields
import numpy as np

SCHEMA_VERSION = 'mtmc-fall/1'


def snapshot(value, shape, confidence=False):
    a = np.array(value, dtype=np.float32, copy=True)
    if a.shape != shape or not np.isfinite(a).all():
        raise ValueError(f'Expected finite {shape}, got {a.shape}')
    if confidence and np.any((a < 0) | (a > 1)):
        raise ValueError('Confidence must be in [0,1]')
    a.setflags(write=False)
    return a


@dataclass(frozen=True)
class TrackedPose:
    frame_index: int
    timestamp: float
    camera_id: int
    global_id: int
    local_id: int
    epoch: int
    detector_index: int
    image_size: tuple[int, int]
    bbox: np.ndarray
    detection_confidence: float
    keypoints: np.ndarray
    confidence: np.ndarray
    partial: bool = False
    occluded: bool = False

    def __post_init__(self):
        if self.frame_index < 0 or self.camera_id not in (1, 2) or self.global_id < 1:
            raise ValueError('Invalid frame/camera/global ID')
        if self.local_id < 0 or self.epoch < 0 or self.detector_index < 0:
            raise ValueError('Invalid local identity/detector index')
        if not np.isfinite(self.timestamp) or self.timestamp < 0:
            raise ValueError('Invalid timestamp')
        if len(self.image_size) != 2 or min(self.image_size) <= 0:
            raise ValueError('Invalid image size')
        if not np.isfinite(self.detection_confidence) or not 0 <= self.detection_confidence <= 1:
            raise ValueError('Invalid detection confidence')
        object.__setattr__(self, 'image_size', tuple(self.image_size))
        for name, shape, conf in [('bbox', (4,), False), ('keypoints', (17, 2), False), ('confidence', (17,), True)]:
            object.__setattr__(self, name, snapshot(getattr(self, name), shape, conf))

    @property
    def slot(self):
        return self.global_id, self.camera_id

    @property
    def identity(self):
        return self.global_id, self.camera_id, self.local_id, self.epoch, self.image_size

    def to_dict(self):
        return {f.name: (getattr(self, f.name).tolist() if isinstance(getattr(self, f.name), np.ndarray)
                         else getattr(self, f.name)) for f in fields(self)}


@dataclass(frozen=True)
class PoseSequence:
    poses: tuple[TrackedPose, ...]
    fps: float

    def __post_init__(self):
        object.__setattr__(self, 'poses', tuple(self.poses))
        if not self.poses or not np.isfinite(self.fps) or self.fps <= 0:
            raise ValueError('A nonempty sequence and positive fps are required')
        first = self.poses[0]
        for i, p in enumerate(self.poses):
            if p.identity != first.identity or p.frame_index != first.frame_index + i:
                raise ValueError('Sequence must contain one contiguous identity/camera segment')
            if not np.isclose(p.timestamp, p.frame_index / self.fps, rtol=0, atol=1e-6):
                raise ValueError('Sequence timestamp does not match source frame')

    @property
    def name(self):
        p = self.poses[0]
        return f'g{p.global_id}_c{p.camera_id}_f{p.frame_index}'


class SequenceCollector:
    """Keep camera histories separate; never interpolate across missing observations."""
    def __init__(self, fps):
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError('Invalid fps')
        self.fps, self.next_frame = float(fps), 0
        self.active, self.completed = {}, []

    def append_frame(self, index, poses):
        poses = list(poses)
        if index != self.next_frame:
            raise ValueError('Append each source frame once in order, including empty frames')
        slots = [p.slot for p in poses]
        if len(set(slots)) != len(slots) or any(p.frame_index != index for p in poses):
            raise ValueError('Duplicate identity/camera slot or incorrect frame')
        for p in poses:
            if not np.isclose(p.timestamp, index / self.fps, rtol=0, atol=1e-6):
                raise ValueError('Incorrect timestamp')
        incoming = {p.slot: p for p in poses}
        for slot in list(self.active):
            if slot not in incoming or incoming[slot].identity != self.active[slot][-1].identity:
                self.completed.append(PoseSequence(tuple(self.active.pop(slot)), self.fps))
        for p in poses:
            self.active.setdefault(p.slot, []).append(p)
        self.next_frame += 1

    def finish(self):
        for poses in self.active.values():
            self.completed.append(PoseSequence(tuple(poses), self.fps))
        self.active.clear()
        return sorted(self.completed, key=lambda s: (s.poses[0].global_id, s.poses[0].camera_id, s.poses[0].frame_index))
