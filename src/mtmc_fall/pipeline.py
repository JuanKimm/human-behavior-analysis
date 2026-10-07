from __future__ import annotations
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
import json
import os
import time
from .contracts import SCHEMA_VERSION, SequenceCollector
from .environment import verify_environment
from .fall import OfflineFallAnalyzer, select_camera_results
from .output import write_json, write_jsonl_row, save_sequence, render_video
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
    """Offline sequential frames -> committed identities -> per-camera poses -> fall decisions."""
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
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    legacy = load_tracker(root)
    weight_path, _ = legacy.prepare_weights(config.reid_weights, True, legacy.REID_SHA256)
    os.environ['YOLO_AUTOINSTALL'] = 'false'; os.environ['YOLO_OFFLINE'] = 'true'
    run_dir = Path(config.output_root) / datetime.now(timezone.utc).strftime('run_%Y%m%d_%H%M%S_%f')
    run_dir.mkdir(parents=True, exist_ok=False)
    summary = {'schema_version': SCHEMA_VERSION, 'status': 'failed', 'settings': {k: str(v) for k, v in asdict(config).items()},
               'versions': versions, 'preprocessing': preprocessing, 'frames': 0, 'sequences': 0,
               'source_sha256': legacy.sha256_file(root / 'mtmct_delayed_realtime_v4-2-2.py'),
               'weights_sha256': {str(p): legacy.sha256_file(p) for p in paths[2:]}}
    reader = None; start = time.perf_counter()
    try:
        import torch
        from ultralytics import YOLO
        from ultralytics.cfg import DEFAULT_CFG_DICT
        from preprocessing.videopose3d import VideoPose3DLifter
        from runtime.engine import FallRiskInferenceEngine
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
        vp = cfg.get('videopose3d', {})
        lifter = VideoPose3DLifter(root / cfg['paths']['videopose3d_weights'], device=config.device,
                    filter_widths=vp.get('filter_widths'), causal=vp.get('causal', False), channels=vp.get('channels', 1024))
        if lifter.receptive_field != int(vp['receptive_field_frames']):
            raise ValueError('VideoPose3D receptive field mismatch')
        analyzer = OfflineFallAnalyzer(lifter, FallRiskInferenceEngine(root, device=config.device))
        precision = {'quantize': 32} if 'quantize' in DEFAULT_CFG_DICT else {'half': False}
        args = SimpleNamespace(imgsz=config.imgsz, device=config.device, yolo_precision=precision)
        collector = SequenceCollector(reader.fps); all_poses = []
        with (run_dir / 'events.jsonl').open('w', encoding='utf-8') as events, (run_dir / 'tracked_poses.jsonl').open('w', encoding='utf-8') as tracking:
            adapter = TrackingAdapter(legacy, model, encoder, calibration, reader.fps, args,
                                      event_sink=lambda row: write_jsonl_row(events, row))
            def emit(item):
                packet, rows, lookahead, tail = item
                collector.append_frame(packet.sequence, rows); all_poses.extend(rows)
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
        sequences = collector.finish(); summary['sequences'] = len(sequences)
        camera_rows = []
        for i, sequence in enumerate(sequences, 1):
            print(f'[fall] {i}/{len(sequences)} {sequence.name}', flush=True)
            skeleton, result, timeline = analyzer.analyze(sequence)
            save_sequence(run_dir / 'sequences', sequence, skeleton, result)
            camera_rows.extend(timeline)
        risks = select_camera_results(camera_rows)
        with (run_dir / 'risks.jsonl').open('w', encoding='utf-8') as stream:
            for row in risks:
                write_jsonl_row(stream, dict(schema_version=SCHEMA_VERSION, **row))
        if config.write_video:
            render_video(config.videos, run_dir / 'combined_fall.mp4', all_poses, risks, legacy, config.display_width)
        summary.update(status='ok', global_ids=sorted({p.global_id for p in all_poses}),
                       valid_risk_rows=sum(r['status'] == 'ok' for r in risks), tracking_timings=adapter.timings.report())
        return run_dir
    except BaseException as exc:
        summary['failure'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        if reader is not None:
            reader.close()
        summary['elapsed_seconds'] = time.perf_counter() - start
        write_json(run_dir / 'summary.json', summary)
