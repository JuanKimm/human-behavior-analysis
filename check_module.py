from pathlib import Path
import sys, json, hashlib, argparse
ROOT=Path(__file__).resolve().parent; SRC=ROOT/'src'
if str(SRC) not in sys.path: sys.path.insert(0,str(SRC))

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def verify_release_manifest():
    mf=ROOT/'RELEASE_MANIFEST.json'; dig=ROOT/'RELEASE_MANIFEST.sha256'
    if not mf.exists() or not dig.exists():
        raise RuntimeError('RELEASE_MANIFEST.json/.sha256 missing; run tools/build_release_manifest.py')
    expected=dig.read_text(encoding='ascii').split()[0]
    actual=sha(mf)
    if expected!=actual: raise RuntimeError('RELEASE_MANIFEST.json detached SHA-256 mismatch')
    obj=json.loads(mf.read_text(encoding='utf-8')); errors=[]
    for e in obj['files']:
        p=ROOT/e['path']
        if not p.exists(): errors.append(f"MISS {e['path']}")
        elif p.stat().st_size!=e['size_bytes'] or sha(p)!=e['sha256']: errors.append(f"DIFF {e['path']}")
    if errors:
        for x in errors[:50]: print(x)
        raise RuntimeError(f'release manifest mismatch count={len(errors)}')
    print(f"OK   release manifest files={len(obj['files'])}")

def verify_protected_models():
    final=ROOT/'PHASE1_FINAL_MANIFEST.json'
    if not final.exists(): raise RuntimeError('PHASE1_FINAL_MANIFEST.json missing')
    protected=json.loads(final.read_text(encoding='utf-8'))['protected_models']; bad=[]
    for e in protected:
        p=ROOT/e['path']; actual=sha(p) if p.exists() else None
        if actual!=e['sha256'] or not e.get('matches_original'): bad.append(e['path'])
    if bad: raise RuntimeError(f'protected model checksum mismatch: {bad}')
    print(f'OK   protected model checksums={len(protected)}')

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--video',default=None); ap.add_argument('--device',default='auto'); ap.add_argument('--allow-unvalidated-environment',action='store_true',help='engineering diagnostics only; official raw-video runs must not use this override'); a=ap.parse_args()
    cfg=json.loads((ROOT/'config/module_config.json').read_text(encoding='utf-8'))
    from runtime.environment_contract import verify_preprocessing_environment
    env_contract=verify_preprocessing_environment(ROOT,allow_unvalidated=a.allow_unvalidated_environment)
    print('preprocessing_environment:',env_contract['status'],env_contract.get('profile_id'), 'override=' + str(env_contract.get('override_used')))
    print('Project:',ROOT); print('package_version:',cfg.get('versions',{}).get('package_version'))
    for k,p in cfg['paths'].items():
        f=ROOT/p
        if not f.exists(): raise FileNotFoundError(f'Missing required asset {k}: {p}')
        print(f'OK   {k}: {p}')
    verify_release_manifest(); verify_protected_models()
    import numpy as np
    from runtime.engine import FallRiskInferenceEngine
    eng=FallRiskInferenceEngine(ROOT,device=a.device)
    for n,want in [(15,'insufficient_length'),(16,'partial'),(63,'partial'),(64,'ok')]:
        r=eng.infer_skeleton_sequence(np.zeros((n,17,3),np.float32),source=f'check_{n}')
        if r.status!=want: raise AssertionError((n,r.status,want))
    r=eng.infer_skeleton_sequence(np.zeros((64,17,3),np.float32),source='check_module')
    assert len(r.tcn)==7 and len(r.stds)==1 and len(r.dbn)==1
    print('OK   64F core forward: TCN=7 STDS=1 DBN=1')
    if a.video:
        from runtime.video_pipeline import VideoInferenceModule
        result,_=VideoInferenceModule(ROOT,device=a.device).run_video(a.video)
        print(f'OK   video E2E frames={result.frame_count} status={result.status}')
    print('PASS check_module')
if __name__=='__main__': main()
