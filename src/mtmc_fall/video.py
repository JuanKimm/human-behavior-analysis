from __future__ import annotations
import math
import time


class PairedVideoReader:
    """Already synchronized, equal-FPS inputs. Decode each source frame exactly once."""
    def __init__(self, paths, packet_type, cv2_module=None):
        if cv2_module is None:
            import cv2 as cv2_module
        self.cv2, self.packet_type = cv2_module, packet_type
        self.caps, self.info, self.sequence = {}, {}, 0
        try:
            if len(paths) != 2:
                raise ValueError('Exactly two videos are required')
            for cam, path in enumerate(paths, 1):
                cap = self.cv2.VideoCapture(str(path)); self.caps[cam] = cap
                if not cap.isOpened():
                    raise ValueError(f'Cannot open camera {cam}: {path}')
                fps = float(cap.get(self.cv2.CAP_PROP_FPS))
                count = float(cap.get(self.cv2.CAP_PROP_FRAME_COUNT))
                w, h = (int(cap.get(k)) for k in (self.cv2.CAP_PROP_FRAME_WIDTH, self.cv2.CAP_PROP_FRAME_HEIGHT))
                if not math.isfinite(fps) or fps <= 0 or not math.isfinite(count) or count <= 0 or count != int(count) or min(w, h) <= 0:
                    raise ValueError(f'Invalid video metadata: camera {cam}')
                self.info[cam] = dict(fps=fps, frame_count=int(count), width=w, height=h)
            a, b = self.info[1], self.info[2]
            if abs(a['fps'] - b['fps']) > 1e-6 or a['frame_count'] != b['frame_count']:
                raise ValueError('Inputs must have identical FPS and frame count; no time correction is performed')
            self.fps, self.frame_count = a['fps'], a['frame_count']
        except BaseException:
            self.close(); raise

    def next(self):
        pairs = {cam: cap.read() for cam, cap in self.caps.items()}
        if self.sequence == self.frame_count:
            if any(ok for ok, _ in pairs.values()):
                raise RuntimeError('Decoded frame count exceeds metadata')
            return None
        if not all(ok for ok, _ in pairs.values()):
            raise RuntimeError(f'Premature decode failure at frame {self.sequence}')
        frames = {cam: frame for cam, (_, frame) in pairs.items()}
        for cam, frame in frames.items():
            if frame.shape[:2] != (self.info[cam]['height'], self.info[cam]['width']):
                raise RuntimeError('Video resolution changed')
        index = self.sequence; self.sequence += 1
        timestamp = index / self.fps
        return self.packet_type(sequence=index, timestamp=timestamp, frames=frames,
                                indices={1: index, 2: index}, times={1: timestamp, 2: timestamp},
                                fresh={1: True, 2: True}, arrival=time.perf_counter())

    def close(self):
        for cap in self.caps.values():
            cap.release()
