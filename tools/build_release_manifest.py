from __future__ import annotations
import hashlib, json
from pathlib import Path
from datetime import datetime, timezone
ROOT=Path(__file__).resolve().parents[1]
EXCLUDE_NAMES={'RELEASE_MANIFEST.json','RELEASE_MANIFEST.sha256'}
EXCLUDE_PARTS={'__pycache__','.git','.pytest_cache'}
EXCLUDE_PREFIXES=('outputs/',)

def sha(p):
    h=hashlib.sha256();
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def main():
    files=[]
    for p in sorted(ROOT.rglob('*')):
        if not p.is_file(): continue
        rel=p.relative_to(ROOT).as_posix()
        if p.name in EXCLUDE_NAMES or any(x in p.parts for x in EXCLUDE_PARTS) or rel.startswith(EXCLUDE_PREFIXES): continue
        files.append({'path':rel,'size_bytes':p.stat().st_size,'sha256':sha(p)})
    cfg=json.loads((ROOT/'config/module_config.json').read_text(encoding='utf-8'))
    obj={'manifest_schema_version':'1.0','created_utc':datetime.now(timezone.utc).isoformat(),'versions':cfg.get('versions',{}),'files':files}
    out=ROOT/'RELEASE_MANIFEST.json'; out.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    digest=sha(out); (ROOT/'RELEASE_MANIFEST.sha256').write_text(digest+'  RELEASE_MANIFEST.json\n',encoding='ascii')
    print(f'WROTE {out.name}: files={len(files)} sha256={digest}')
if __name__=='__main__': main()
