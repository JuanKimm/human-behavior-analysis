from __future__ import annotations
import hashlib, json, platform, sys
from pathlib import Path
from datetime import datetime, timezone


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def hash_if_exists(path: str | Path) -> str | None:
    p = Path(path)
    return sha256_file(p) if p.is_file() else None


def project_versions(root: str | Path) -> dict:
    root = Path(root)
    cfg = json.loads((root / 'config/module_config.json').read_text(encoding='utf-8'))
    versions = dict(cfg.get('versions', {}))
    versions.setdefault('package_version', cfg.get('module_version'))
    versions.setdefault('model_bundle_id', 'repeat01_seed42')
    versions.setdefault('dataset_version', 'unified3d_labels_v1_provisional')
    versions.setdefault('split_id', 'repeat01_seed42')
    versions.setdefault('output_schema_version', '2.0')
    return versions


def model_hashes(root: str | Path) -> dict:
    root = Path(root)
    cfg = json.loads((root / 'config/module_config.json').read_text(encoding='utf-8'))
    keys = ('yolo_weights','videopose3d_weights','tcn_checkpoint','stds_checkpoint','dbn_checkpoint')
    return {k: {'path': cfg['paths'][k], 'sha256': hash_if_exists(root / cfg['paths'][k])} for k in keys}


def dataset_hashes(root: str | Path, gt_npz: str | Path | None = None) -> dict:
    root = Path(root)
    split = root / 'reference/repeat01_config/split.json'
    out = {'split_json': {'path': str(split.relative_to(root)), 'sha256': hash_if_exists(split)}}
    if gt_npz:
        p = Path(gt_npz)
        out['ground_truth_npz'] = {'path': str(p), 'sha256': hash_if_exists(p)}
    elif (root / 'unified_3d_all_envs.npz').exists():
        p = root / 'unified_3d_all_envs.npz'
        out['ground_truth_npz'] = {'path': str(p.relative_to(root)), 'sha256': hash_if_exists(p)}
    return out


def basic_environment() -> dict:
    info = {
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'python': sys.version,
        'python_executable': sys.executable,
        'platform': platform.platform(),
        'machine': platform.machine(),
    }
    try:
        import numpy as np
        info['numpy'] = np.__version__
    except Exception:
        pass
    try:
        import torch
        info.update(torch=torch.__version__, cuda_available=bool(torch.cuda.is_available()), torch_cuda_version=torch.version.cuda)
    except Exception:
        pass
    try:
        import cv2
        info['opencv'] = cv2.__version__
    except Exception:
        pass
    try:
        import ultralytics
        info['ultralytics'] = ultralytics.__version__
    except Exception:
        pass
    try:
        import torchvision
        info['torchvision'] = torchvision.__version__
    except Exception:
        pass
    try:
        import sklearn
        info['scikit_learn'] = sklearn.__version__
    except Exception:
        pass
    return info
