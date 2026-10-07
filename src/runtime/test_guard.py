from __future__ import annotations
import json
from pathlib import Path


def split_of_video(root: str | Path, video_key: str) -> str | None:
    p=Path(root)/'reference/repeat01_config/split.json'
    if not p.exists(): return None
    d=json.loads(p.read_text(encoding='utf-8'))
    for split in ('train','val','test'):
        if video_key in set(map(str,d.get(f'{split}_videos',[]))): return split
    return None


def enforce_test_protection(root: str | Path, video_key: str, allow_test=False):
    split=split_of_video(root,video_key)
    if split=='test' and not allow_test:
        raise PermissionError(f'{video_key} is in seed42 test set. Final test is locked until Gate-3 approval; pass allow_test=True only for the approved final evaluation.')
    return split
