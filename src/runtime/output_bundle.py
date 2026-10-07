from __future__ import annotations
import re
from pathlib import Path


def sanitize_key(value: str) -> str:
    value = value.replace('::','__')
    value = re.sub(r'[^0-9A-Za-z가-힣._()\-]+', '_', value).strip('._ ')
    return value or 'input'


def allocate_run_dir(output_root: str | Path, video_key: str) -> Path:
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    base = sanitize_key(video_key)
    for i in range(1, 1000000):
        p = root / f'{base}_run_{i:03d}'
        try:
            p.mkdir()
            return p
        except FileExistsError:
            continue
    raise RuntimeError(f'Could not allocate run directory for {base}')
