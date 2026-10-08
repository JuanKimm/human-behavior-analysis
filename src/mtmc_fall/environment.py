from importlib import metadata
import json

VERSIONS = {'numpy': '2.2.6', 'opencv-python': '4.11.0.86', 'ultralytics': '8.4.70',
            'torch': '2.13.0', 'torchvision': '0.28.0', 'scikit-learn': '1.8.0', 'lap': '0.5.12'}
TORCHREID_COMMIT = 'f8cd150fdf77e8d9e1ed143b7f308c2c609ded50'


def verify_environment():
    versions, mismatches = {}, []
    for package, expected in VERSIONS.items():
        try:
            actual = metadata.version(package)
        except metadata.PackageNotFoundError:
            actual = 'missing'
        versions[package] = actual
        if actual.split('+')[0] != expected:
            mismatches.append(f'{package}: expected {expected}, got {actual}')
    try:
        dist = metadata.distribution('torchreid')
        direct = json.loads(dist.read_text('direct_url.json') or '{}')
        commit = direct.get('vcs_info', {}).get('commit_id')
        versions['torchreid'] = {'version': dist.version, 'commit': commit}
    except metadata.PackageNotFoundError:
        commit = None
    if commit != TORCHREID_COMMIT:
        mismatches.append(f'torchreid must be installed from commit {TORCHREID_COMMIT}')
    if mismatches:
        raise RuntimeError('Integration environment mismatch: ' + '; '.join(mismatches))
    return versions
