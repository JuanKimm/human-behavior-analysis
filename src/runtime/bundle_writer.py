from __future__ import annotations
import json, os, sys
from pathlib import Path
from datetime import datetime, timezone
from .output_bundle import allocate_run_dir
from .provenance import sha256_file, hash_if_exists, model_hashes, dataset_hashes, project_versions, basic_environment
from .evaluation import make_timeline, write_timeline_csv, compute_metrics, transition_summary, write_transition_csv
from .profile import result_for_profile
from .overlay import write_overlay_video


def _write_json(path, obj):
    Path(path).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def write_run_bundle(root, inference_dict, input_path, output_root, video_key, mode='predict', profile='research', gt=None, gt_npz=None, command=None, create_overlay=False, runtime_info=None):
    root=Path(root); input_path=Path(input_path) if input_path else None
    run_dir=allocate_run_dir(output_root,video_key); evidence=run_dir/'_evidence'; evidence.mkdir()
    if gt is not None and len(gt.labels) != int(inference_dict['frame_count']):
        raise ValueError(f'GT length {len(gt.labels)} != inference frame_count {inference_dict["frame_count"]}')
    timeline=make_timeline(inference_dict,None if gt is None else gt.labels)
    write_timeline_csv(run_dir/'frame_timeline.csv',timeline)
    trans=transition_summary(timeline,inference_dict.get('fps')) if gt is not None else []
    write_transition_csv(run_dir/'transition_summary.csv',trans)
    metrics={'mode':mode,'ground_truth_available':gt is not None,'runtime':runtime_info or {},'models':compute_metrics(timeline) if gt is not None else {}}
    _write_json(run_dir/'metrics_summary.json',metrics)
    result_view=result_for_profile(inference_dict,profile); result_view['mode']=mode; result_view['video_key']=video_key
    _write_json(run_dir/'result.json',result_view)
    overlay_status='not_requested'
    if create_overlay and input_path and input_path.suffix.lower() in ('.mp4','.avi','.mov','.mkv'):
        try:
            write_overlay_video(input_path,run_dir/f'{input_path.stem}_overlay.mp4',timeline,profile=profile); overlay_status='ok'
        except Exception as e:
            overlay_status=f'failed:{type(e).__name__}:{e}'
    env=basic_environment(); env['inference_environment']=inference_dict.get('environment'); _write_json(evidence/'environment.json',env)
    manifest={
      'created_utc':datetime.now(timezone.utc).isoformat(),'mode':mode,'profile':profile,'video_key':video_key,
      'command':command or ' '.join(sys.argv),'input':None if input_path is None else {'path':str(input_path),'sha256':hash_if_exists(input_path)},
      'versions':project_versions(root),'models':model_hashes(root),'datasets':dataset_hashes(root,gt_npz),
      'config_sha256':hash_if_exists(root/'config/module_config.json'),'inference_status':inference_dict.get('status'),
      'model_status':inference_dict.get('model_status'),'overlay_status':overlay_status,'runtime':runtime_info or {},
    }
    _write_json(evidence/'run_manifest.json',manifest)
    required=['frame_timeline.csv','transition_summary.csv','metrics_summary.json','result.json','_evidence/environment.json','_evidence/run_manifest.json']
    if create_overlay: required.append(f'{input_path.stem}_overlay.mp4')
    missing=[x for x in required if not (run_dir/x).exists()]
    status={'status':'ok' if not missing and not overlay_status.startswith('failed') else 'failed','missing_files':missing,'overlay_status':overlay_status,
            'inference_status':inference_dict.get('status'),'required_files':required}
    _write_json(evidence/'validation_status.json',status)
    outputs={}
    for p in sorted(run_dir.rglob('*')):
        if p.is_file() and p.name not in ('output_hashes.json','validation_status.json'): outputs[str(p.relative_to(run_dir))]=sha256_file(p)
    _write_json(evidence/'output_hashes.json',outputs)
    status['output_hashes_sha256']=sha256_file(evidence/'output_hashes.json'); _write_json(evidence/'validation_status.json',status)
    return {'run_dir':str(run_dir),'validation_status':status,'manifest':manifest}
