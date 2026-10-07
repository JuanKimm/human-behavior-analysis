from __future__ import annotations
import csv, json, math
from pathlib import Path
import numpy as np

GT_LABELS = ('Normal','Precursor','Danger')
TCN_LABELS = ('Non-Danger','Danger')


def _projection(rows, n, label_key='label', status_key='status'):
    labels = [None] * n
    status = [None] * n
    decision = [None] * n
    probs = [None] * n
    for row in sorted(rows, key=lambda x: int(x['window_end'])):
        d = int(row['window_end'])
        if d >= n:
            continue
        nxt = n
        # fill only until next decision; later rows overwrite their own onward interval.
        for j in range(d, n):
            if decision[j] is not None:
                break
            labels[j] = row.get(label_key)
            status[j] = row.get(status_key)
            decision[j] = d
            probs[j] = row.get('probabilities')
    return labels, status, decision, probs


def _projection_custom(rows, n, label_key):
    labels = [None] * n; status=[None]*n; decision=[None]*n
    for row in sorted(rows, key=lambda x:int(x['window_end'])):
        d=int(row['window_end'])
        if d>=n: continue
        for j in range(d,n):
            if decision[j] is not None: break
            labels[j]=row.get(label_key); status[j]=row.get('status'); decision[j]=d
    return labels,status,decision


def _fill_hold_last(rows, n, label_key='label', prob_key='probabilities'):
    # forward-fill from each decision until the next decision, never before first decision
    labels=[None]*n; status=[None]*n; decision=[None]*n; probs=[None]*n
    ordered=sorted((r for r in rows if int(r['window_end']) < n), key=lambda r:int(r['window_end']))
    for idx,r in enumerate(ordered):
        d=int(r['window_end']); end=(int(ordered[idx+1]['window_end']) if idx+1<len(ordered) else n)
        for j in range(d, min(end,n)):
            labels[j]=r.get(label_key); status[j]=r.get('status'); decision[j]=d; probs[j]=r.get(prob_key)
    return labels,status,decision,probs


def make_timeline(inference: dict, gt_labels: np.ndarray | None = None):
    n=int(inference['frame_count']); fps=inference.get('fps')
    tcn_l,tcn_s,tcn_d,tcn_p=_fill_hold_last(inference.get('tcn',[]),n)
    pre_l,pre_s,pre_d,pre_p=_fill_hold_last(inference.get('stds',[]),n,'pre_crf_label','pre_crf_probabilities')
    post_l,post_s,post_d,_=_fill_hold_last(inference.get('stds',[]),n,'post_crf_label','_none')
    dbn_l,dbn_s,dbn_d,dbn_p=_fill_hold_last(inference.get('dbn',[]),n)
    rows=[]
    for i in range(n):
        gt=None if gt_labels is None else GT_LABELS[int(gt_labels[i])]
        rows.append({
            'frame_idx':i,
            'timestamp_sec':None if not fps else i/float(fps),
            'gt_state':gt,
            'gt_tcn':None if gt_labels is None else ('Danger' if int(gt_labels[i])==2 else 'Non-Danger'),
            'tcn_label':tcn_l[i], 'tcn_status':tcn_s[i], 'tcn_decision_frame':tcn_d[i], 'tcn_probabilities':None if tcn_p[i] is None else json.dumps(tcn_p[i],separators=(',',':')), 
            'stds_pre_label':pre_l[i], 'stds_pre_status':pre_s[i], 'stds_pre_decision_frame':pre_d[i], 'stds_pre_probabilities':None if pre_p[i] is None else json.dumps(pre_p[i],separators=(',',':')), 
            'stds_post_label':post_l[i], 'stds_post_status':post_s[i], 'stds_post_decision_frame':post_d[i],
            'dbn_label':dbn_l[i], 'dbn_status':dbn_s[i], 'dbn_decision_frame':dbn_d[i], 'dbn_probabilities':None if dbn_p[i] is None else json.dumps(dbn_p[i],separators=(',',':')), 
        })
    return rows


def write_timeline_csv(path: str | Path, rows):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    fields=list(rows[0].keys()) if rows else ['frame_idx']
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)


def _metrics(y_true, y_pred, labels):
    yt=np.asarray(y_true); yp=np.asarray(y_pred)
    cm=np.zeros((len(labels),len(labels)),dtype=int)
    mapidx={v:i for i,v in enumerate(labels)}
    for a,b in zip(yt,yp): cm[mapidx[a],mapidx[b]] += 1
    per={}
    fvals=[]
    for i,label in enumerate(labels):
        tp=cm[i,i]; fp=cm[:,i].sum()-tp; fn=cm[i,:].sum()-tp
        p=float(tp/(tp+fp)) if tp+fp else 0.0
        r=float(tp/(tp+fn)) if tp+fn else 0.0
        f=float(2*p*r/(p+r)) if p+r else 0.0
        per[label]={'precision':p,'recall':r,'f1':f,'support':int(cm[i,:].sum())}; fvals.append(f)
    return {'accuracy':float((yt==yp).mean()) if len(yt) else None,'macro_f1':float(np.mean(fvals)) if fvals else None,
            'labels':list(labels),'confusion_matrix':cm.tolist(),'per_class':per,'evaluated_frames':int(len(yt))}


def compute_metrics(timeline):
    specs={
      'tcn':('gt_tcn','tcn_label','tcn_status',TCN_LABELS),
      'stds_pre':('gt_state','stds_pre_label','stds_pre_status',GT_LABELS),
      'stds_post':('gt_state','stds_post_label','stds_post_status',GT_LABELS),
      'dbn':('gt_state','dbn_label','dbn_status',GT_LABELS),
    }
    out={}
    for name,(g,p,st,labels) in specs.items():
        pairs=[(r[g],r[p]) for r in timeline if r.get(g) is not None and r.get(p) in labels and r.get(st)=='ok']
        out[name]=_metrics([a for a,_ in pairs],[b for _,b in pairs],labels)
    return out


def _transitions(seq):
    out=[]; prev=None
    for frame,label in seq:
        if label is None: continue
        if prev is None: prev=label; continue
        if label != prev:
            out.append((frame,prev,label)); prev=label
    return out


def transition_summary(timeline, fps=None):
    gt=_transitions([(r['frame_idx'],r['gt_state']) for r in timeline])
    model_specs={'stds_pre':'stds_pre_label','stds_post':'stds_post_label','dbn':'dbn_label'}
    rows=[]
    for model,col in model_specs.items():
        pred=_transitions([(r['frame_idx'],r[col]) for r in timeline])
        for gf,fr,to in gt:
            candidates=[x for x in pred if x[2]==to]
            chosen=min(candidates,key=lambda x:abs(x[0]-gf)) if candidates else None
            pf=None if chosen is None else int(chosen[0]); delay=None if pf is None else pf-int(gf)
            rows.append({'model':model,'from_state':fr,'to_state':to,'gt_transition_frame':int(gf),
                         'pred_transition_frame':pf,'delay_frames':delay,
                         'gt_transition_time_sec':None if not fps else float(gf)/float(fps),
                         'pred_transition_time_sec':None if not fps or pf is None else float(pf)/float(fps),
                         'delay_sec':None if not fps or delay is None else float(delay)/float(fps),
                         'status':'miss' if pf is None else ('early' if delay<0 else ('late' if delay>0 else 'on_time'))})
    # TCN binary danger onset
    gt_t=_transitions([(r['frame_idx'],r['gt_tcn']) for r in timeline]); pr_t=_transitions([(r['frame_idx'],r['tcn_label']) for r in timeline])
    for gf,fr,to in gt_t:
        if to!='Danger': continue
        c=[x for x in pr_t if x[2]=='Danger']; chosen=min(c,key=lambda x:abs(x[0]-gf)) if c else None
        pf=None if chosen is None else int(chosen[0]); delay=None if pf is None else pf-int(gf)
        rows.append({'model':'tcn','from_state':fr,'to_state':to,'gt_transition_frame':int(gf),'pred_transition_frame':pf,
                     'delay_frames':delay,'gt_transition_time_sec':None if not fps else gf/float(fps),
                     'pred_transition_time_sec':None if not fps or pf is None else pf/float(fps),
                     'delay_sec':None if not fps or delay is None else delay/float(fps),
                     'status':'miss' if pf is None else ('early' if delay<0 else ('late' if delay>0 else 'on_time'))})
    return rows


def write_transition_csv(path, rows):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    fields=['model','from_state','to_state','gt_transition_frame','pred_transition_frame','delay_frames','gt_transition_time_sec','pred_transition_time_sec','delay_sec','status']
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
