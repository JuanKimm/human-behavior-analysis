from __future__ import annotations
import json
from pathlib import Path


def current_preprocessing_versions() -> dict:
    out={'numpy':None,'opencv':None,'ultralytics':None}
    try:
        import numpy as np
        out['numpy']=str(np.__version__)
    except Exception:
        pass
    try:
        import cv2
        out['opencv']=str(cv2.__version__)
    except Exception:
        pass
    try:
        import ultralytics
        out['ultralytics']=str(ultralytics.__version__)
    except Exception:
        pass
    return out


def evaluate_preprocessing_contract(expected: dict, current: dict) -> dict:
    mismatches=[]
    for k,want in expected.items():
        got=current.get(k)
        if str(got) != str(want):
            mismatches.append({'package':k,'expected':str(want),'actual':None if got is None else str(got)})
    return {'status':'ok' if not mismatches else 'mismatch','expected':dict(expected),'current':dict(current),'mismatches':mismatches}


def verify_preprocessing_environment(project_root: str | Path, allow_unvalidated: bool=False) -> dict:
    root=Path(project_root)
    cfg=json.loads((root/'config/module_config.json').read_text(encoding='utf-8'))
    contract=cfg.get('environment_contract',{})
    expected=contract.get('required_import_versions',{})
    current=current_preprocessing_versions()
    result=evaluate_preprocessing_contract(expected,current)
    result['profile_id']=contract.get('profile_id')
    result['scope']=contract.get('scope','raw_video')
    result['override_used']=bool(allow_unvalidated and result['status']!='ok')
    if result['status']!='ok' and not allow_unvalidated:
        detail=', '.join(f"{x['package']} expected={x['expected']} actual={x['actual']}" for x in result['mismatches'])
        raise RuntimeError(
            'Unvalidated raw-video preprocessing environment. '
            + detail
            + '. Install the validated versions in requirements.txt or use an explicitly documented engineering override.'
        )
    return result
