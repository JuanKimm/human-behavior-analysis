from __future__ import annotations
import json
from pathlib import Path
from collections import defaultdict
import numpy as np
from .video import PairedVideoReader

# OpenCV uses BGR. Orange/red belong only to original-video fall status.
RISK_COLORS = {'Precursor': (0, 165, 255), 'Danger': (0, 0, 255)}
TRACK_COLORS = (
    (0, 200, 0), (255, 180, 0), (255, 80, 80), (220, 80, 180),
    (200, 200, 0), (100, 210, 150), (255, 150, 200), (180, 210, 100),
)


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def write_jsonl_row(stream, value):
    stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + '\n')


def save_video_result(directory, camera_id, inference, arrays):
    directory.mkdir(parents=True, exist_ok=True)
    # Original frame order; no ID/gap segmentation or added skeleton processing.
    np.savez_compressed(directory / f'camera{camera_id}.npz',
        skeletons_3d=arrays['skeletons_3d'], confidence=arrays['confidence'],
        detected=arrays['detected'], fps=inference['fps'])
    write_json(directory / f'camera{camera_id}.json', inference)


# Panel layout. All colors are OpenCV BGR.
PANEL_WIDTH = 1120
PANEL_MARGIN = 12
PANEL_OPACITY = 0.76
PANEL_HEIGHT = 316
STATE_COLORS = {
    'Normal': (65, 195, 85), 'Non-Danger': (65, 195, 85),
    'Precursor': RISK_COLORS['Precursor'], 'Danger': RISK_COLORS['Danger'],
}
UNKNOWN_COLOR = (90, 90, 90)


class _ModelDisplay:
    """Project saved windows with the original hold-last rule; no inference."""
    def __init__(self, inference, key, label_key, probability_key, labels):
        self.rows = sorted(inference.get(key, []), key=lambda row: int(row['window_end']))
        self.ends = np.array([int(row['window_end']) for row in self.rows], dtype=np.int64)
        self.label_key, self.probability_key, self.labels = label_key, probability_key, labels
        self.frame_count = int(inference['frame_count'])

    def at(self, frame_index):
        i = int(np.searchsorted(self.ends, frame_index, side='right')) - 1
        return None if i < 0 else self.rows[i]

    def label(self, row):
        if row is None or row.get('status') != 'ok':
            return None
        label = row.get(self.label_key)
        return label if label in self.labels else None

    def timeline(self, width, height):
        # Full offline history, including future decisions, as in the reference.
        strip = np.full((height, width, 3), UNKNOWN_COLOR, dtype=np.uint8)
        n = self.frame_count
        for i, row in enumerate(self.rows):
            start = max(0, int(row['window_end']))
            end = min(n, int(self.rows[i + 1]['window_end']) if i + 1 < len(self.rows) else n)
            if end <= start or start >= n:
                continue
            x1, x2 = round(start * width / n), round(end * width / n)
            strip[:, x1:x2] = STATE_COLORS.get(self.label(row), UNKNOWN_COLOR)
        return strip


class _VideoPanel:
    def __init__(self, inference, camera_id, expected_frames, expected_fps):
        self.frame_count = int(inference['frame_count'])
        self.fps = float(inference['fps'])
        if self.frame_count != expected_frames or abs(self.fps - expected_fps) > 1e-6:
            raise ValueError('Saved fall results do not match the rendered video')
        self.camera_id = camera_id
        self.models = [
            _ModelDisplay(inference, 'tcn', 'label', 'probabilities', ('Non-Danger', 'Danger')),
            _ModelDisplay(inference, 'stds', 'pre_crf_label', 'pre_crf_probabilities', ('Normal', 'Precursor', 'Danger')),
            _ModelDisplay(inference, 'dbn', 'label', 'probabilities', ('Normal', 'Precursor', 'Danger')),
        ]
        self.strips = [model.timeline(984, 14) for model in self.models]

    @staticmethod
    def _text(cv2, frame, value, pos, scale=.52, color=(235, 235, 235), thickness=1):
        cv2.putText(frame, str(value), pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)

    def draw(self, frame, frame_index):
        import cv2
        # Draw at a stable design resolution, then fit the complete panel to the
        # available source frame. A same-size dark background remains translucent.
        h, w = frame.shape[:2]
        margin = min(PANEL_MARGIN, max(0, min(h, w) // 20))
        width = min(PANEL_WIDTH, w - 2 * margin)
        scale = min(width / 1120., (h - 2 * margin) / PANEL_HEIGHT)
        width, height = max(1, round(1120 * scale)), max(1, round(PANEL_HEIGHT * scale))
        roi = frame[margin:margin + height, margin:margin + width]
        base = cv2.resize(roi, (1120, PANEL_HEIGHT), interpolation=cv2.INTER_LINEAR)
        panel = cv2.addWeighted(base, 1. - PANEL_OPACITY, np.zeros_like(base), PANEL_OPACITY, 0.)
        text = lambda value, pos, **kw: self._text(cv2, panel, value, pos, **kw)
        text(f'Frame {frame_index + 1}/{self.frame_count}   Time {frame_index / self.fps:.2f}s / {self.frame_count / self.fps:.2f}s',
             (12, 25), scale=.64, thickness=2)
        text(f'C{self.camera_id} | First detection | windows: 0-based', (672, 25), scale=.48)
        titles = ('Skeleton-TCN', 'ST-DS-Transformer', 'DBN')
        for index, (title, model) in enumerate(zip(titles, self.models)):
            x, top, card_w = 12 + index * 368, 40, 360
            cv2.rectangle(panel, (x, top), (x + card_w - 1, 221), (140, 140, 140), 1)
            text(title, (x + 10, 64), scale=.62, thickness=2)
            row = model.at(frame_index)
            label = model.label(row)
            if label is None:
                status = 'Not available yet' if row is None else str(row.get('status') or 'N/A')
                text(status, (x + 10, 94), color=(185, 185, 185))
            else:
                color = STATE_COLORS[label]
                cv2.rectangle(panel, (x + 10, 75), (x + 163, 102), color, -1)
                text(label, (x + 16, 96), scale=.55, color=(20, 20, 20), thickness=2)
            if row is not None:
                text(f"window: {row['window_start']}-{row['window_end']}", (x + 176, 94), scale=.43)
            if index == 1:
                post = (row.get('post_crf_label') if label is not None else None) or 'N/A'
                text(f'Bars/label: pre-CRF | CRF: {post}', (x + 10, 121), scale=.41)
            probs = None if row is None or label is None else row.get(model.probability_key)
            if probs is not None:
                probs = np.asarray(probs, dtype=np.float64)
                if probs.shape != (len(model.labels),) or not np.isfinite(probs).all() or np.any((probs < 0) | (probs > 1)):
                    probs = None
            for j, name in enumerate(model.labels):
                y = 145 + j * 28
                text(name, (x + 10, y), scale=.48)
                bar_x, bar_w = x + 115, 176
                cv2.rectangle(panel, (bar_x, y - 13), (bar_x + bar_w - 1, y + 2), (70, 70, 70), -1)
                if probs is None:
                    value = 'N/A'
                else:
                    p = float(probs[j]); filled = round(p * bar_w)
                    if filled:
                        cv2.rectangle(panel, (bar_x, y - 13), (bar_x + filled - 1, y + 2), STATE_COLORS[name], -1)
                    value = f'{p:.4f}'
                text(value, (x + 299, y), scale=.43)
        for i, (short, model, strip) in enumerate(zip(('TCN', 'ST-DS pre', 'DBN'), self.models, self.strips)):
            y = 232 + 25 * i
            text(short, (14, y + 13), scale=.46)
            panel[y:y + 14, 124:1108] = strip
            # Frame cells cover [0,N); center the cursor on the current frame.
            cursor = 124 + min(983, int((frame_index + .5) * 984 / self.frame_count))
            cv2.line(panel, (cursor, y - 2), (cursor, y + 16), (255, 255, 255), 2)
        text('Timeline: complete video | Gray: unavailable | White: current frame', (124, 309), scale=.4)
        roi[:] = cv2.resize(panel, (width, height), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)


def render_video(paths, output_path, poses, risks, legacy, display_width=1280):
    import cv2
    by_frame = defaultdict(list)
    for p in poses:
        by_frame[p.frame_index].append(p)
    reader = PairedVideoReader(paths, legacy.Packet)
    writer = None
    try:
        # Read the already saved original model outputs once. No model is run
        # here, and no global-ID/camera-selection logic is introduced.
        panels = {}
        for camera_id in (1, 2):
            result_path = Path(output_path).parent / 'fall' / f'camera{camera_id}.json'
            with result_path.open(encoding='utf-8') as stream:
                inference = json.load(stream)
            panels[camera_id] = _VideoPanel(inference, camera_id, reader.frame_count, reader.fps)
        while (packet := reader.next()) is not None:
            for p in by_frame[packet.sequence]:
                frame = packet.frames[p.camera_id]
                color = TRACK_COLORS[(int(p.global_id) - 1) % len(TRACK_COLORS)]
                x1, y1, x2, y2 = np.rint(p.bbox).astype(int)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f'G{p.global_id}', (x1, max(20, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, .6, color, 2)
            for camera_id, frame in packet.frames.items():
                panels[camera_id].draw(frame, packet.sequence)
            combined = legacy.combine_frames([packet.frames[1], packet.frames[2]], display_width)
            if writer is None:
                writer = legacy.make_writer(output_path, reader.fps, (combined.shape[1], combined.shape[0]))
            writer.write(combined)
    finally:
        reader.close()
        if writer is not None:
            writer.release()
