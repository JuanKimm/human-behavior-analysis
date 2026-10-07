from __future__ import annotations
import copy


def result_for_profile(result: dict, profile: str) -> dict:
    if profile not in ('research','product'):
        raise ValueError("profile must be 'research' or 'product'")
    out = copy.deepcopy(result)
    out['presentation_profile'] = profile
    if profile == 'product':
        out.pop('stds', None)
        for row in out.get('dbn', []):
            for key in ('t4','sp','sd','stds_probabilities'):
                row.pop(key, None)
    return out
