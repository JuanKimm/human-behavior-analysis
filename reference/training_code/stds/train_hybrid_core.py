# -*- coding: utf-8 -*-
"""
Frozen TCN + ST-DS-Transformer + CRF.
매 epoch last.pt atomic save + resume.
"""

from __future__ import annotations

import csv
import json
import os
import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from .model_hybrid import HybridSTDSTransformer
from .data_unified import (
    CLASS_NAMES, NUM_CLASSES, build_sequence_clips, describe_segment_lengths,
    load_unified, precursor_segment_lengths, split_videos_from_json,
)

EPOCHS = 80
PATIENCE = 15
BATCH_SIZE = 16
EVAL_BATCH_SIZE = 32
LR = 5e-4
WEIGHT_DECAY = 1e-4
WINDOW = 64
STRIDE_COARSE = 8
STRIDE_DENSE = 2
FOCAL_GAMMA = 2.0
CRF_LOSS_WEIGHT = 1.0
FOCAL_LOSS_WEIGHT = 0.5
GRAD_CLIP = 1.0


class SequenceDS(Dataset):
    def __init__(self, x, y, frame_ids):
        self.x, self.y, self.frame_ids = x, y, frame_ids

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return (
            torch.from_numpy(self.x[i]),
            torch.from_numpy(self.y[i]).long(),
            torch.from_numpy(self.frame_ids[i]).long(),
        )


class FrameFocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("alpha", alpha if alpha is not None else None)

    def forward(self, emissions, target):
        logits = emissions.reshape(-1, emissions.shape[-1])
        y = target.reshape(-1)
        logp = F.log_softmax(logits, dim=1)
        logp_t = logp.gather(1, y.unsqueeze(1)).squeeze(1)
        p_t = logp_t.exp()
        loss = -((1.0 - p_t) ** self.gamma) * logp_t
        if self.alpha is not None:
            loss = loss * self.alpha.gather(0, y)
        return loss.mean()


def require_cuda():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA를 사용할 수 없습니다.")
    return torch.device("cuda")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def atomic_torch_save(obj, path):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def compute_metrics(t, p):
    t, p = np.asarray(t), np.asarray(p)
    labels = list(range(NUM_CLASSES))
    pp = precision_score(t, p, average=None, labels=labels, zero_division=0)
    rr = recall_score(t, p, average=None, labels=labels, zero_division=0)
    ff = f1_score(t, p, average=None, labels=labels, zero_division=0)
    out = {
        "accuracy": float(accuracy_score(t, p)),
        "macro_f1": float(f1_score(t, p, average="macro", zero_division=0)),
        "cm": confusion_matrix(t, p, labels=labels).tolist(),
        "per_class": {},
    }
    for i, name in enumerate(CLASS_NAMES):
        out["per_class"][name] = {
            "precision": float(pp[i]), "recall": float(rr[i]), "f1": float(ff[i])
        }
    return out


@torch.no_grad()
def evaluate_unique_frames(model, loader, device):
    model.eval()
    votes, truth = {}, {}
    for x, y, frame_ids in loader:
        emissions = model(x.to(device, non_blocking=True))
        pred = model.decode(emissions=emissions).cpu().numpy()
        y_np, fid_np = y.numpy(), frame_ids.numpy()
        for b in range(pred.shape[0]):
            for t in range(pred.shape[1]):
                fid = int(fid_np[b, t])
                if fid not in votes:
                    votes[fid] = np.zeros(NUM_CLASSES, dtype=np.int64)
                    truth[fid] = int(y_np[b, t])
                votes[fid][int(pred[b, t])] += 1

    ids = sorted(votes)
    yt = np.asarray([truth[i] for i in ids], dtype=np.int64)
    yp = np.asarray([np.argmax(votes[i]) for i in ids], dtype=np.int64)
    out = compute_metrics(yt, yp)
    out["n_unique_frames"] = int(len(ids))
    return out


def train_stds(
    unified_path, tcn_ckpt_path, split_json, run_dir, seed=42, patience=PATIENCE
):
    if int(patience) <= 0: raise ValueError('patience must be a positive integer')
    patience = int(patience)
    device = require_cuda()
    set_seed(seed)

    unified_path = Path(unified_path)
    tcn_ckpt_path = Path(tcn_ckpt_path)
    split_json = Path(split_json)
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    data = load_unified(unified_path)
    tr_v, va_v, te_v, split = split_videos_from_json(data, split_json)

    xtr, ytr, _, ftr = build_sequence_clips(
        data, tr_v, window=WINDOW, stride=STRIDE_COARSE,
        mixed=True, stride_dense=STRIDE_DENSE
    )
    xva, yva, _, fva = build_sequence_clips(
        data, va_v, window=WINDOW, stride=STRIDE_COARSE,
        mixed=False, stride_dense=STRIDE_DENSE
    )

    if len(xtr) == 0 or len(xva) == 0:
        raise RuntimeError("ST-DS train/val window가 비어 있습니다.")

    counts = np.bincount(ytr.reshape(-1), minlength=NUM_CLASSES).astype(np.float64)
    weights = counts.sum() / (NUM_CLASSES * np.maximum(counts, 1.0))

    config = {
        "stage": "Frozen TCN + ST-DS-Transformer + CRF",
        "seed": int(seed),
        "split_json": str(split_json.resolve()),
        "split_ratio": split["requested_ratio"],
        "tcn_checkpoint": str(tcn_ckpt_path.resolve()),
        "tcn_frozen": True,
        "train_windows": int(len(xtr)),
        "val_windows": int(len(xva)),
        "reserved_test_videos": int(len(te_v)),
        "window": WINDOW,
        "stride_coarse": STRIDE_COARSE,
        "stride_dense": STRIDE_DENSE,
        "epochs": EPOCHS,
        "patience": patience,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    with (run_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    tr_dl = DataLoader(
        SequenceDS(xtr, ytr, ftr), batch_size=BATCH_SIZE, shuffle=True,
        num_workers=0, pin_memory=True
    )
    va_dl = DataLoader(
        SequenceDS(xva, yva, fva), batch_size=EVAL_BATCH_SIZE, shuffle=False,
        num_workers=0, pin_memory=True
    )

    model = HybridSTDSTransformer(
        tcn_ckpt_path=tcn_ckpt_path,
        in_channels=xtr.shape[-1],
        num_frames=WINDOW,
        num_joints=xtr.shape[2],
        num_classes=NUM_CLASSES,
        use_crf=True,
        freeze_tcn=True,
    ).to(device)

    alpha = torch.tensor(weights, dtype=torch.float32, device=device)
    focal = FrameFocalLoss(alpha=alpha, gamma=FOCAL_GAMMA).to(device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_path = run_dir / "best.pt"
    last_path = run_dir / "last.pt"
    log_path = run_dir / "train_log.csv"

    start_epoch = 1
    best_f1 = -1.0
    best_epoch = 0
    bad = 0

    if last_path.exists():
        ckpt = torch.load(last_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        best_f1 = float(ckpt.get("best_val_macro_f1", -1.0))
        best_epoch = int(ckpt.get("best_epoch", 0))
        bad = int(ckpt.get("bad_epochs", 0))
        start_epoch = int(ckpt["epoch"]) + 1
        print(f"[STDS RESUME] epoch {start_epoch}부터 재개")

    if not log_path.exists():
        with log_path.open("w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(
                ["epoch", "train_loss", "crf_loss", "focal_loss",
                 "val_accuracy", "val_macro_f1", "val_precursor_recall", "is_best"]
            )

    writer = SummaryWriter(log_dir=str(run_dir / "tensorboard"))
    started = time.time()

    if bad < patience and start_epoch <= EPOCHS:
        for epoch in range(start_epoch, EPOCHS + 1):
            model.train()
            total_losses, crf_losses, focal_losses = [], [], []

            for xb, yb, _ in tr_dl:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)

                emissions = model(xb)
                crf_loss = model.crf_nll(emissions, yb)
                focal_loss = focal(emissions, yb)
                loss = CRF_LOSS_WEIGHT * crf_loss + FOCAL_LOSS_WEIGHT * focal_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable, GRAD_CLIP)
                optimizer.step()

                total_losses.append(float(loss.item()))
                crf_losses.append(float(crf_loss.item()))
                focal_losses.append(float(focal_loss.item()))

            scheduler.step()
            vm = evaluate_unique_frames(model, va_dl, device)
            score = vm["macro_f1"]
            prec_r = vm["per_class"]["전조"]["recall"]
            is_best = score > best_f1

            if is_best:
                best_f1 = score
                best_epoch = epoch
                bad = 0
                atomic_torch_save(
                    {
                        "model_state_dict": {
                            k: v.detach().cpu().clone()
                            for k, v in model.state_dict().items()
                        },
                        "best_epoch": int(best_epoch),
                        "best_val_macro_f1": float(best_f1),
                        "seed": int(seed),
                        "tcn_frozen": True,
                    },
                    best_path,
                )
                with (run_dir / "best_val_metrics.json").open("w", encoding="utf-8") as f:
                    json.dump(vm, f, indent=2, ensure_ascii=False)
            else:
                bad += 1

            avg_total = float(np.mean(total_losses))
            avg_crf = float(np.mean(crf_losses))
            avg_focal = float(np.mean(focal_losses))

            atomic_torch_save(
                {
                    "epoch": int(epoch),
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_val_macro_f1": float(best_f1),
                    "best_epoch": int(best_epoch),
                    "bad_epochs": int(bad),
                    "seed": int(seed),
                },
                last_path,
            )

            with log_path.open("a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(
                    [epoch, f"{avg_total:.6f}", f"{avg_crf:.6f}",
                     f"{avg_focal:.6f}", f"{vm['accuracy']:.6f}",
                     f"{score:.6f}", f"{prec_r:.6f}", int(is_best)]
                )

            writer.add_scalar("Loss/train_total", avg_total, epoch)
            writer.add_scalar("F1/val_macro", score, epoch)

            print(
                f"[STDS {epoch:02d}] loss={avg_total:.5f} "
                f"val_macroF1={score:.4f} precR={prec_r:.4f} "
                f"({time.time()-started:.0f}s)"
                + (" <- BEST" if is_best else "")
            )

            if bad >= patience:
                print(f"[STDS] early stop @ epoch {epoch}")
                break

    writer.close()

    if not best_path.exists():
        raise RuntimeError("ST-DS best.pt가 없습니다.")

    summary = {
        "stage": "stds",
        "status": "done",
        "seed": int(seed),
        "best_epoch": int(best_epoch),
        "best_val_macro_f1": float(best_f1),
        "external_test_evaluated": False,
    }
    with (run_dir / "stage_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    (run_dir / "DONE").write_text("OK\n", encoding="utf-8")
    return summary
