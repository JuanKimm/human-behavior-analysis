from __future__ import annotations
import json, os, tempfile
from pathlib import Path

def write_json_atomic(path, payload, overwrite=False):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite: raise FileExistsError(f'Output exists; pass --overwrite: {path}')
    content = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)
    fd, temp = tempfile.mkstemp(prefix=f'.{path.name}.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            f.write(content); f.flush(); os.fsync(f.fileno())
        if path.exists() and not overwrite: raise FileExistsError(path)
        os.replace(temp, path)
    finally:
        if os.path.exists(temp): os.unlink(temp)
