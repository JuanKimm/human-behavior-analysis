from __future__ import annotations
from pathlib import Path
import json
import threading
import numpy as np
import torch
import torch.nn.functional as F

from tcn.model import SkeletonTCN
from stds.model_hybrid import HybridSTDSTransformer
from preprocessing.skeleton import validate_skeleton_sequence
from .dbn_filter import StatefulDBNFilter
from .env_info import resolve_device, environment_info
from .schemas import TCNResult, STDSResult, DBNResult, InferenceResult

class FallRiskInferenceEngine:
    DEFAULT_TCN_LABELS = ('Non-Danger','Danger')
    DEFAULT_STATE_LABELS = ('Normal','Precursor','Danger')

    def __init__(self, project_root: str | Path, config_path: str | Path | None = None, device: str | None = None):
        self.root = Path(project_root).resolve()
        config_path = Path(config_path) if config_path else self.root/'config/module_config.json'
        self.config = json.loads(config_path.read_text(encoding='utf-8'))
        runtime = self.config.get('runtime', {})
        contract = self.config.get('checkpoint_contract', {'tcn_window':16,'tcn_stride':8,'stds_window':64,'stds_stride':16,'dbn_stride':16})
        for key, expected in contract.items():
            if key in runtime and runtime.get(key) != expected:
                raise ValueError(f'config runtime.{key}={runtime.get(key)!r}; checkpoint contract requires {expected}')
        self.tcn_labels = tuple(runtime.get('tcn_labels', self.DEFAULT_TCN_LABELS))
        self.state_labels = tuple(runtime.get('stds_labels', runtime.get('dbn_labels', self.DEFAULT_STATE_LABELS)))
        self.dbn_labels = tuple(runtime.get('dbn_labels', self.DEFAULT_STATE_LABELS))
        if len(self.tcn_labels) != 2 or len(self.state_labels) != 3 or len(self.dbn_labels) != 3:
            raise ValueError('TCN labels must have 2 classes and STDS/DBN labels must have 3 classes')
        qcfg=self.config.get('pose_quality',{})
        for key in ('min_detected_ratio','min_valid_joint_ratio','confidence_threshold'):
            value=qcfg.get(key,0.5 if key != 'confidence_threshold' else 0.25)
            if not isinstance(value,(int,float)) or not np.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f'pose_quality.{key} must be finite and in [0,1]')
        resolved, _ = resolve_device(device, self.config.get('device','auto'))
        self.device = torch.device(resolved)
        self.device_name = resolved
        self.environment = environment_info(resolved)
        self.module_version = self.config.get('versions',{}).get('package_version', self.config.get('module_version'))

        paths = self.config['paths']
        self.tcn_ckpt = self.root/paths['tcn_checkpoint']
        self.stds_ckpt = self.root/paths['stds_checkpoint']
        self.dbn_ckpt = self.root/paths['dbn_checkpoint']

        self.tcn = SkeletonTCN(in_channels=3, num_joints=17, num_classes=2).to(self.device)
        tcn_obj = torch.load(self.tcn_ckpt, map_location='cpu', weights_only=True)
        tcn_state = tcn_obj['model_state_dict'] if isinstance(tcn_obj,dict) and 'model_state_dict' in tcn_obj else tcn_obj
        self.tcn.load_state_dict(tcn_state, strict=True)
        self.tcn.eval()

        self.stds = HybridSTDSTransformer(
            tcn_ckpt_path=self.tcn_ckpt,
            in_channels=3, num_frames=64, num_joints=17, num_classes=3,
            use_crf=True, freeze_tcn=True,
        ).to(self.device)
        stds_obj = torch.load(self.stds_ckpt, map_location='cpu', weights_only=True)
        stds_state = stds_obj['model_state_dict'] if isinstance(stds_obj,dict) and 'model_state_dict' in stds_obj else stds_obj
        self.stds.load_state_dict(stds_state, strict=True)
        self.stds.eval()
        internal_tcn = getattr(self.stds, 'tcn', None)
        if internal_tcn is None:
            raise RuntimeError('STDS checkpoint does not expose its embedded TCN for synchronization validation')
        external_state=self.tcn.state_dict(); internal_state=internal_tcn.state_dict()
        if external_state.keys()!=internal_state.keys(): raise RuntimeError('Standalone and embedded TCN state keys differ')
        mismatch=[k for k in external_state if not torch.equal(external_state[k].detach().cpu(),internal_state[k].detach().cpu())]
        if mismatch: raise RuntimeError(f'Standalone TCN checkpoint and embedded STDS TCN are out of sync; first keys: {mismatch[:5]}')
        self.dbn = StatefulDBNFilter(self.dbn_ckpt)
        self.dbn.LABELS = self.dbn_labels
        self._lock = threading.RLock()

        self.tcn_window = int(runtime['tcn_window']); self.tcn_stride = int(runtime['tcn_stride'])
        self.stds_window = int(runtime['stds_window']); self.stds_stride = int(runtime['stds_stride'])
        self.dbn_stride = int(runtime.get('dbn_stride', self.stds_stride))

    def reset(self):
        with self._lock:
            self.dbn.reset()

    @torch.no_grad()
    def _tcn_one(self, clip: np.ndarray):
        x = torch.from_numpy(clip).unsqueeze(0).permute(0,3,1,2).contiguous().to(self.device)
        p = F.softmax(self.tcn(x), dim=1)[0].detach().cpu().numpy()
        idx = int(np.argmax(p))
        return self.tcn_labels[idx], p

    @torch.no_grad()
    def _stds_outputs(self, clip: np.ndarray):
        x = torch.from_numpy(clip).unsqueeze(0).to(self.device)
        out = self.stds.get_pre_crf_outputs(x)
        emissions = out['emissions']
        pre_last = out['stds_precrf_probs'][0,-1].detach().cpu().numpy().astype(np.float64)
        pre_idx = int(np.argmax(pre_last)); pre_label = self.state_labels[pre_idx]
        post_path_t = self.stds.decode(emissions=emissions)[0].detach().cpu().numpy().astype(int)
        post_path = post_path_t.tolist(); post_label = self.state_labels[int(post_path_t[-1])]
        t4_probs = out['tcn_chunk_probs'][0,3].detach().cpu().numpy().astype(np.float64)
        eps=1e-6
        stds_for_dbn=np.clip(pre_last,eps,1-eps); t4_probs=np.clip(t4_probs,eps,1-eps)
        t4=float(np.log(t4_probs[1]/t4_probs[0])); sp=float(np.log(stds_for_dbn[1]/stds_for_dbn[0])); sd=float(np.log(stds_for_dbn[2]/stds_for_dbn[0]))
        return pre_last, pre_label, post_label, post_path, t4, sp, sd

    def infer_skeleton_sequence(self, skeletons, source: str='skeleton', fps: float | None=None,
                                detected=None, confidence=None, diagnostics=None) -> InferenceResult:
        x = validate_skeleton_sequence(skeletons)
        if fps is not None and (not np.isfinite(float(fps)) or float(fps) <= 0):
            raise ValueError(f'fps must be positive and finite: {fps}')
        if detected is not None:
            detected = np.asarray(detected).astype(bool).reshape(-1)
            if len(detected) != len(x): raise ValueError(f'detected length {len(detected)} != skeleton length {len(x)}')
        if confidence is not None:
            confidence=np.asarray(confidence,dtype=np.float32)
            if confidence.shape != (len(x),17) or not np.isfinite(confidence).all() or np.any((confidence<0)|(confidence>1)):
                raise ValueError(f'confidence must be finite [0,1] shape (N,17): {confidence.shape}')
        with self._lock:
            return self._infer_locked(x, source, fps, detected, confidence, diagnostics)

    def _infer_locked(self, x, source, fps, detected, confidence, diagnostics):
        self.dbn.reset()
        qcfg=self.config.get('pose_quality',{})
        det_threshold=float(qcfg.get('min_detected_ratio',0.5)); joint_threshold=float(qcfg.get('min_valid_joint_ratio',0.5)); conf_threshold=float(qcfg.get('confidence_threshold',0.25))
        def quality(s,e):
            if detected is None: return None
            d=detected[s:e]; ratio=float(d.mean()) if len(d) else 0.0; longest=run=0
            for present in d:
                run=0 if present else run+1; longest=max(longest,run)
            vjr=None if confidence is None else (float((confidence[s:e]>=conf_threshold).mean()) if e>s else 0.0)
            insufficient=ratio<det_threshold or (vjr is not None and vjr<joint_threshold)
            return {'detected_ratio':ratio,'valid_joint_ratio':vjr,'longest_consecutive_missed_frames':int(longest),
                    'confidence_threshold':conf_threshold,'insufficient_pose':bool(insufficient),'thresholds_provisional':True}

        tcn=[]
        for s in range(0,len(x)-self.tcn_window+1,self.tcn_stride):
            e=s+self.tcn_window; label,p=self._tcn_one(x[s:e]); q=quality(s,e)
            st='insufficient_pose' if q and q['insufficient_pose'] else 'ok'
            tcn.append(TCNResult(s,e-1,label,p.tolist(),None if q is None else q['detected_ratio'],st,q))

        stds=[]; dbn=[]
        for s in range(0,len(x)-self.stds_window+1,self.stds_stride):
            e=s+self.stds_window; q=quality(s,e)
            pre,pre_label,post_label,post_path,t4,sp,sd=self._stds_outputs(x[s:e])
            st='insufficient_pose' if q and q['insufficient_pose'] else 'ok'
            stds.append(STDSResult(s,e-1,pre_label,pre.tolist(),post_label,post_path,None if q is None else q['detected_ratio'],st,q))
            if q and q['insufficient_pose']:
                self.dbn.reset(); dbn.append(DBNResult(s,e-1,'insufficient_pose',[],t4,sp,sd,pre.tolist(),q['detected_ratio'],'insufficient_pose',q))
            else:
                dr=self.dbn.update(t4,sp,sd); dbn.append(DBNResult(s,e-1,dr['label'],dr['probabilities'],t4,sp,sd,pre.tolist(),None if q is None else q['detected_ratio'],'ok',q))

        n=len(x); detected_ratio=None if detected is None else (float(detected.mean()) if n else 0.0)
        global_joint_ratio=None if confidence is None else (float((confidence>=conf_threshold).mean()) if len(confidence) else 0.0)
        global_insufficient=detected_ratio is not None and (detected_ratio<det_threshold or (global_joint_ratio is not None and global_joint_ratio<joint_threshold))
        any_window_insufficient = any(r.status != 'ok' for r in tcn) or any(r.status != 'ok' for r in stds) or any(r.status != 'ok' for r in dbn)
        if n == 0:
            top='no_frames'
        elif n < self.tcn_window:
            top='insufficient_length'
        elif n < self.stds_window:
            top='partial'
        elif global_insufficient:
            top='insufficient_pose'
        elif any_window_insufficient:
            top='partial'
        else:
            top='ok'
        model_status={
            'tcn':'ok' if n>=self.tcn_window else 'insufficient_length',
            'stds':'ok' if n>=self.stds_window else 'insufficient_length',
            'dbn':'ok' if n>=self.stds_window else 'insufficient_length',
        }
        if global_insufficient and n>=self.tcn_window: model_status['tcn']='insufficient_pose'
        if global_insufficient and n>=self.stds_window: model_status['stds']=model_status['dbn']='insufficient_pose'
        quality_all={'source':'video_detection' if detected is not None else 'unavailable_3d_input','detected_ratio':detected_ratio,'valid_joint_ratio':global_joint_ratio,
                     'thresholds':{'min_detected_ratio':det_threshold,'min_valid_joint_ratio':joint_threshold,'confidence_threshold':conf_threshold},
                     'thresholds_provisional':detected is not None,'dbn_low_quality_policy':'reset state and emit insufficient_pose with no classification probabilities'}
        return InferenceResult(source=source,frame_count=n,fps=fps,tcn=tcn,stds=stds,dbn=dbn,module_version=self.module_version,
                               device=self.device_name,environment=self.environment,detected_ratio=detected_ratio,status=top,model_status=model_status,
                               pose_quality=quality_all,diagnostics=list(diagnostics or []))
