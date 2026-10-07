# -*- coding: utf-8 -*-
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from stds.data_unified import load_unified
from stds.model_hybrid import HybridSTDSTransformer
from .data_dbn import WindowDataset, build_window_records, video_indices_from_split
from .metrics import CLASS_NAMES, compute_metrics, save_metrics
from .model_dbn import FinalGaussianDBN

DBN_WINDOW = 64
DBN_STRIDE = 16
EVAL_BATCH_SIZE = 64
PROB_EPS = 1e-6


def require_cuda():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA를 사용할 수 없습니다.")
    return torch.device("cuda")


def load_stds_model(stds_ckpt_path, tcn_ckpt_path, device, in_channels=3, num_joints=17):
    model = HybridSTDSTransformer(
        tcn_ckpt_path=tcn_ckpt_path,
        in_channels=in_channels,
        num_frames=64,
        num_joints=num_joints,
        num_classes=3,
        use_crf=True,
        freeze_tcn=True,
    ).to(device)

    obj = torch.load(stds_ckpt_path, map_location="cpu", weights_only=True)
    state = obj["model_state_dict"] if isinstance(obj, dict) and "model_state_dict" in obj else obj
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def _clip(p):
    return np.clip(np.asarray(p, dtype=np.float64), PROB_EPS, 1.0 - PROB_EPS)


@torch.no_grad()
def collect_observation_rows(data, records, model, device):
    dl = DataLoader(
        WindowDataset(data, records, window=DBN_WINDOW),
        batch_size=EVAL_BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True
    )
    rows = [None] * len(records)

    for x, _, ridx in dl:
        out = model.get_pre_crf_outputs(x.to(device, non_blocking=True))
        stds_probs = out["stds_precrf_probs"].detach().cpu().numpy()
        tcn_probs = out["tcn_chunk_probs"].detach().cpu().numpy()

        stds_last = _clip(stds_probs[:, -1, :])      # frame 64, CRF 전
        t4_probs = _clip(tcn_probs[:, 3, :])         # frames 49~64

        t4 = np.log(t4_probs[:, 1] / t4_probs[:, 0])
        sp = np.log(stds_last[:, 1] / stds_last[:, 0])
        sd = np.log(stds_last[:, 2] / stds_last[:, 0])

        for b, row_idx in enumerate(ridx.numpy().astype(np.int64)):
            r = records[int(row_idx)]
            vi = int(r.video_idx)
            rows[int(row_idx)] = {
                "video_idx": vi,
                "video_name": str(data["video_names"][vi]),
                "env_name": str(data["env_names"][vi]),
                "window_abs_start": int(r.abs_start),
                "window_abs_end": int(r.abs_end),
                "target_label": int(r.target_label),
                "target_name": CLASS_NAMES[int(r.target_label)],
                "tcn_t4_non_danger_prob": float(t4_probs[b, 0]),
                "tcn_t4_danger_prob": float(t4_probs[b, 1]),
                "stds_normal_prob": float(stds_last[b, 0]),
                "stds_precursor_prob": float(stds_last[b, 1]),
                "stds_danger_prob": float(stds_last[b, 2]),
                "T4": float(t4[b]),
                "S_P": float(sp[b]),
                "S_D": float(sd[b]),
            }
    return rows


def save_rows_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    header = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        w.writerows(rows)


def train_dbn_stage(project_root, repeat_dir, split_json):
    project_root = Path(project_root).resolve()
    repeat_dir = Path(repeat_dir).resolve()
    device = require_cuda()

    data_path = project_root / "data" / "unified_3d_all_envs.npz"
    if not data_path.exists(): data_path = project_root / "unified_3d_all_envs.npz"
    data = load_unified(data_path)
    train_videos, _, test_videos = video_indices_from_split(split_json, data)

    records = build_window_records(
        data, train_videos, window=DBN_WINDOW, stride=DBN_STRIDE
    )
    model = load_stds_model(
        repeat_dir / "02_stds" / "best.pt",
        repeat_dir / "01_tcn" / "best.pt",
        device,
        in_channels=data["skeletons"].shape[-1],
        num_joints=data["skeletons"].shape[-2],
    )

    rows = collect_observation_rows(data, records, model, device)
    run_dir = repeat_dir / "03_dbn"
    run_dir.mkdir(parents=True, exist_ok=True)
    save_rows_csv(run_dir / "dbn_train_observations.csv", rows)

    dbn = FinalGaussianDBN(
        alpha=0.1, cov_reg=1e-3, var_floor=1e-6, environment_balanced=True
    ).fit(rows)
    dbn.save(run_dir / "final_dbn.npz")

    summary = {
        "stage": "dbn",
        "status": "done",
        "train_videos": int(len(train_videos)),
        "reserved_test_videos": int(len(test_videos)),
        "train_windows": int(len(records)),
        "features": ["T4", "S_P", "S_D"],
        "stds_source": "pre-CRF softmax at frame64",
        "tcn_source": "frames49-64",
        "covariance": "full",
        "environment_balanced": True,
        "inference": "filtering",
    }
    with (run_dir / "stage_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    (run_dir / "DONE").write_text("OK\n", encoding="utf-8")
    return summary


def final_test_stage(project_root, repeat_dir, split_json):
    project_root = Path(project_root).resolve()
    repeat_dir = Path(repeat_dir).resolve()
    device = require_cuda()

    data_path = project_root / "data" / "unified_3d_all_envs.npz"
    if not data_path.exists(): data_path = project_root / "unified_3d_all_envs.npz"
    data = load_unified(data_path)
    _, _, test_videos = video_indices_from_split(split_json, data)
    records = build_window_records(
        data, test_videos, window=DBN_WINDOW, stride=DBN_STRIDE
    )

    model = load_stds_model(
        repeat_dir / "02_stds" / "best.pt",
        repeat_dir / "01_tcn" / "best.pt",
        device,
        in_channels=data["skeletons"].shape[-1],
        num_joints=data["skeletons"].shape[-2],
    )
    rows = collect_observation_rows(data, records, model, device)

    run_dir = repeat_dir / "04_test"
    run_dir.mkdir(parents=True, exist_ok=True)
    save_rows_csv(run_dir / "test_observations.csv", rows)

    y_true = np.asarray([r["target_label"] for r in rows], dtype=np.int64)

    stds_probs = np.asarray(
        [[r["stds_normal_prob"], r["stds_precursor_prob"], r["stds_danger_prob"]]
         for r in rows], dtype=np.float64
    )
    stds_metrics = compute_metrics(y_true, stds_probs.argmax(axis=1))
    save_metrics(stds_metrics, run_dir / "stds_precrf_metrics")

    dbn = FinalGaussianDBN.load(repeat_dir / "03_dbn" / "final_dbn.npz")
    post = dbn.filter(rows)
    dbn_metrics = compute_metrics(y_true, post.argmax(axis=1))
    save_metrics(dbn_metrics, run_dir / "dbn_metrics")

    result = {
        "test_videos": int(len(test_videos)),
        "test_windows": int(len(records)),
        "unit": "64-frame window, stride16",
        "stds_precrf": stds_metrics,
        "final_dbn": dbn_metrics,
    }
    with (run_dir / "FINAL_RESULT.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    (run_dir / "DONE").write_text("OK\n", encoding="utf-8")
    return result
