from __future__ import annotations
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
import json
import os
import time
from .contracts import SCHEMA_VERSION
from .environment import verify_environment
from .fall import OriginalVideoFallAnalyzer
from .output import write_json, write_jsonl_row, save_video_result, render_video
from .tracking import load_tracker, TrackingAdapter
from .video import PairedVideoReader


@dataclass(frozen=True)
class OfflineConfig:
    root: Path
    videos: tuple[Path, Path]
    yolo_weights: Path
    reid_weights: Path
    output_root: Path
    device: str = 'cpu'
    imgsz: int = 640
    write_video: bool = True
    display_width: int = 1280


def run_offline(config: OfflineConfig):
    """Original MTMCT tracking plus unchanged whole-video fall inference."""
    if config.device != 'cpu' and not config.device.startswith('cuda'):
        raise ValueError('Use cpu or cuda:N')
    if not 256 <= config.imgsz <= 2048 or config.display_width < 64 or config.display_width % 2:
        raise ValueError('Invalid image/display size')
    root = Path(config.root).resolve()
    versions = verify_environment()
    from runtime.environment_contract import verify_preprocessing_environment
    preprocessing = verify_preprocessing_environment(root)
    cfg = json.loads((root / 'config/module_config.json').read_text(encoding='utf-8'))
    paths = [Path(p) for p in config.videos] + [Path(config.yolo_weights), Path(config.reid_weights)]
    paths += [root / cfg['paths'][k] for k in ('videopose3d_weights', 'tcn_checkpoint', 'stds_checkpoint', 'dbn_checkpoint')]
    fall_yolo_weights = root / cfg['paths']['yolo_weights']
    if fall_yolo_weights not in paths:
        paths.append(fall_yolo_weights)
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    legacy = load_tracker(root)
    weight_path, _ = legacy.prepare_weights(config.reid_weights, True, legacy.REID_SHA256)
    os.environ['YOLO_AUTOINSTALL'] = 'false'; os.environ['YOLO_OFFLINE'] = 'true'
    run_dir = Path(config.output_root) / datetime.now(timezone.utc).strftime('run_%Y%m%d_%H%M%S_%f')
    run_dir.mkdir(parents=True, exist_ok=False)
    summary = {'schema_version': SCHEMA_VERSION, 'status': 'failed', 'settings': {k: str(v) for k, v in asdict(config).items()},
               'versions': versions, 'preprocessing': preprocessing, 'frames': 0, 'fall_videos': 0,
               'fall_mode': 'original_first_detection_full_video',
               'fall_yolo_weights': str(fall_yolo_weights),
               'source_sha256': legacy.sha256_file(root / 'mtmct_delayed_realtime_v4-2-2.py'),
               'weights_sha256': {str(p): legacy.sha256_file(p) for p in paths[2:]}}
    reader = None; start = time.perf_counter()
    try:
        import torch
        from ultralytics import YOLO
        from ultralytics.cfg import DEFAULT_CFG_DICT
        if config.device.startswith('cuda') and not torch.cuda.is_available():
            raise RuntimeError('CUDA is not available')
        if config.device == 'cpu':
            torch.set_num_threads(min(4, os.cpu_count() or 1))
        reader = PairedVideoReader(config.videos, legacy.Packet)
        calibration = legacy.Calibration()
        for cam, info in reader.info.items():
            calibration.check_size(cam, (info['width'], info['height']), False)
        summary['videos'] = reader.info
        write_json(run_dir / 'calibration.json', calibration.report)
        model = YOLO(str(config.yolo_weights))
        if model.task != 'pose':
            raise ValueError('A YOLO pose model is required')
        encoder = legacy.FeatureEncoder(weight_path, config.device, torch, half=False)
        precision = {'quantize': 32} if 'quantize' in DEFAULT_CFG_DICT else {'half': False}
        args = SimpleNamespace(imgsz=config.imgsz, device=config.device, yolo_precision=precision)
        all_poses = []
        with (run_dir / 'events.jsonl').open('w', encoding='utf-8') as events, (run_dir / 'tracked_poses.jsonl').open('w', encoding='utf-8') as tracking:
            adapter = TrackingAdapter(legacy, model, encoder, calibration, reader.fps, args,
                                      event_sink=lambda row: write_jsonl_row(events, row))
            def emit(item):
                packet, rows, lookahead, tail = item
                all_poses.extend(rows)
                write_jsonl_row(tracking, dict(schema_version=SCHEMA_VERSION, frame_index=packet.sequence,
                    timestamp=packet.timestamp, lookahead_seconds=lookahead, truncated_tail=tail,
                    poses=[p.to_dict() for p in rows], diagnostics=packet.diagnostics))
                summary['frames'] += 1
            while (packet := reader.next()) is not None:
                for item in adapter.push(packet):
                    emit(item)
            for item in adapter.flush():
                emit(item)
        reader.close(); reader = None
        # Tracking is complete. Release its models before loading the original
        # fall module, whose detector/settings are intentionally kept independent.
        tracking_timings = adapter.timings.report()
        del adapter, model, encoder
        if config.device.startswith('cuda'):
            torch.cuda.empty_cache()
        analyzer = OriginalVideoFallAnalyzer(root, device=config.device)
        risks = []
        for camera_id, video_path in enumerate(config.videos, 1):
            print(f'[fall-original] camera {camera_id}: {video_path}', flush=True)
            info = summary['videos'][camera_id]
            inference, arrays, timeline = analyzer.analyze_video(
                video_path, camera_id, summary['frames'], info['fps'])
            save_video_result(run_dir / 'fall', camera_id, inference, arrays)
            risks.extend(timeline)
            summary['fall_videos'] += 1
        risks.sort(key=lambda row: (row['frame_index'], row['camera_id']))
        with (run_dir / 'risks.jsonl').open('w', encoding='utf-8') as stream:
            for row in risks:
                write_jsonl_row(stream, dict(schema_version='mtmc-fall/original-video/1', **row))
        if config.write_video:
            render_video(config.videos, run_dir / 'combined_fall.mp4', all_poses, risks, legacy, config.display_width)
        summary.update(status='ok', global_ids=sorted({p.global_id for p in all_poses}),
                       valid_risk_rows=sum(r['status'] == 'ok' for r in risks), tracking_timings=tracking_timings)
        return run_dir
    except BaseException as exc:
        summary['failure'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        if reader is not None:
            reader.close()
        summary['elapsed_seconds'] = time.perf_counter() - start
        write_json(run_dir / 'summary.json', summary)
