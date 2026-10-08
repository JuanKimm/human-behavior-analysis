from __future__ import annotations
import json
from collections import defaultdict
import numpy as np
from .video import PairedVideoReader


# OpenCV uses BGR (blue, green, red), not RGB.
# Orange and red are reserved exclusively for risk states.
RISK_COLORS = {
    'Precursor': (0, 165, 255),  # orange
    'Danger': (0, 0, 255),       # red
}
TRACK_COLORS = (
    (0, 200, 0),      # green
    (255, 180, 0),    # sky blue
    (255, 80, 80),    # blue
    (220, 80, 180),   # purple
    (200, 200, 0),    # cyan
    (100, 210, 150),  # mint green
    (255, 150, 200),  # lavender
    (180, 210, 100),  # turquoise
)


def overlay_color(global_id, risk):
    """Use the selected global risk in both views; otherwise use a stable ID color."""
    if risk.get('status') == 'ok' and risk.get('label') in RISK_COLORS:
        return RISK_COLORS[risk['label']]
    return TRACK_COLORS[(int(global_id) - 1) % len(TRACK_COLORS)]


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def write_jsonl_row(stream, value):
    stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + '\n')


def save_sequence(directory, sequence, skeleton, inference):
    first = sequence.poses[0]
    directory.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(directory / (sequence.name + '.npz'),
        keypoints_2d=np.stack([p.keypoints for p in sequence.poses]),
        confidence=np.stack([p.confidence for p in sequence.poses]), skeletons_3d=skeleton,
        frame_indices=np.array([p.frame_index for p in sequence.poses]),
        timestamps=np.array([p.timestamp for p in sequence.poses]),
        global_id=first.global_id, camera_id=first.camera_id, fps=sequence.fps, image_size=first.image_size)
    write_json(directory / (sequence.name + '.json'), dict(global_id=first.global_id, camera_id=first.camera_id,
        source_start_frame=first.frame_index, source_end_frame=sequence.poses[-1].frame_index,
        window_indexing='zero_based_inclusive_relative_to_sequence', inference=inference))


def render_video(paths, output_path, poses, risks, legacy, display_width=1280):
    import cv2
    by_frame = defaultdict(list)
    for p in poses:
        by_frame[p.frame_index].append(p)
    decisions = {(r['frame_index'], r['global_id']): r for r in risks}
    reader = PairedVideoReader(paths, legacy.Packet)
    writer = None
    try:
        while (packet := reader.next()) is not None:
            for p in by_frame[packet.sequence]:
                frame = packet.frames[p.camera_id]
                risk = decisions.get((packet.sequence, p.global_id), {})
                color = overlay_color(p.global_id, risk)
                x1, y1, x2, y2 = np.rint(p.bbox).astype(int)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label = risk.get('label') or 'N/A'
                camera = risk.get('selected_camera_id')
                text = f'G{p.global_id} {label}' + (f' [C{camera}]' if camera is not None else '')
                cv2.putText(frame, text, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, .6, color, 2)
            combined = legacy.combine_frames([packet.frames[1], packet.frames[2]], display_width)
            if writer is None:
                writer = legacy.make_writer(output_path, reader.fps, (combined.shape[1], combined.shape[0]))
            writer.write(combined)
    finally:
        reader.close()
        if writer is not None:
            writer.release()
