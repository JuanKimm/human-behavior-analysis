# -*- coding: utf-8 -*-
"""
Skeleton-TCN 70/15/15 반복 학습 엔진.
매 epoch 종료 시 last.pt를 atomic save하여 중단 후 이어학습 가능.
"""

from __future__ import annotations

import csv
import json
import os
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from .model import SkeletonTCN
from .data_st import (
    CLASS_NAMES,
    NUM_CLASSES,
    indices_from_split,
    load_all_npz,
    prep_xy_for_tcn,
    to_binary_label,
)

EPOCHS = 50
PATIENCE = 10
BATCH_SIZE = 64
EVAL_BATCH_SIZE = 128
LR = 1e-3
WEIGHT_DECAY = 1e-4


class ClipDS(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return torch.from_numpy(self.x[i]), torch.tensor(self.y[i], dtype=torch.long)


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


def compute_metrics(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    labels = list(range(NUM_CLASSES))
    p = precision_score(y_true, y_pred, average=None, labels=labels, zero_division=0)
    r = recall_score(y_true, y_pred, average=None, labels=labels, zero_division=0)
    f = f1_score(y_true, y_pred, average=None, labels=labels, zero_division=0)
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "cm": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "per_class": {},
    }
    for i, name in enumerate(CLASS_NAMES):
        out["per_class"][name] = {
            "precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i])
        }
    return out


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    yt, yp = [], []
    for x, y in loader:
        pred = model(x.to(device, non_blocking=True)).argmax(1).cpu().numpy()
        yt.extend(y.numpy().tolist())
        yp.extend(pred.tolist())
    return compute_metrics(yt, yp)


def train_tcn(data_dir, split_json, run_dir, seed=42, max_channels=3):
    device = require_cuda()
    set_seed(seed)

    data_dir = Path(data_dir)
    split_json = Path(split_json)
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    x, y_raw, vid, _ = load_all_npz(data_dir, max_channels=max_channels)
    y = to_binary_label(y_raw)
    tr_i, va_i, te_i, split = indices_from_split(vid, split_json)

    xtr, ytr = prep_xy_for_tcn(x[tr_i]), y[tr_i]
    xva, yva = prep_xy_for_tcn(x[va_i]), y[va_i]

    counts = np.bincount(ytr, minlength=NUM_CLASSES).astype(np.float64)
    weights = counts.sum() / (NUM_CLASSES * np.maximum(counts, 1.0))

    config = {
        "stage": "TCN",
        "seed": int(seed),
        "split_json": str(split_json.resolve()),
        "split_ratio": split["requested_ratio"],
        "train_clips": int(len(tr_i)),
        "val_clips": int(len(va_i)),
        "reserved_test_clips": int(len(te_i)),
        "window": 16,
        "stride": 4,
        "epochs": EPOCHS,
        "patience": PATIENCE,
        "batch_size": BATCH_SIZE,
        "lr": LR,
        "weight_decay": WEIGHT_DECAY,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    with (run_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    tr_dl = DataLoader(
        ClipDS(xtr, ytr), batch_size=BATCH_SIZE, shuffle=True,
        num_workers=0, pin_memory=True
    )
    va_dl = DataLoader(
        ClipDS(xva, yva), batch_size=EVAL_BATCH_SIZE, shuffle=False,
        num_workers=0, pin_memory=True
    )

    model = SkeletonTCN(
        in_channels=xtr.shape[1], num_joints=17, num_classes=NUM_CLASSES
    ).to(device)
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(weights, dtype=torch.float32, device=device)
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    last_path = run_dir / "last.pt"
    best_path = run_dir / "best.pt"
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
        print(f"[TCN RESUME] epoch {start_epoch}부터 재개")

    if not log_path.exists():
        with log_path.open("w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(
                ["epoch", "train_loss", "val_accuracy", "val_macro_f1", "is_best"]
            )

    writer = SummaryWriter(log_dir=str(run_dir / "tensorboard"))

    # interruption이 early-stop 직후 발생했을 경우 추가 학습 없이 마무리
    if bad < PATIENCE and start_epoch <= EPOCHS:
        for epoch in range(start_epoch, EPOCHS + 1):
            model.train()
            losses = []

            for xb, yb in tr_dl:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(xb), yb)
                loss.backward()
                optimizer.step()
                losses.append(float(loss.item()))

            scheduler.step()
            train_loss = float(np.mean(losses))
            vm = evaluate(model, va_dl, device)
            score = vm["macro_f1"]

            is_best = score > best_f1
            if is_best:
                best_f1 = score
                best_epoch = epoch
                bad = 0
                atomic_torch_save(
                    {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                    best_path,
                )
                with (run_dir / "best_val_metrics.json").open("w", encoding="utf-8") as f:
                    json.dump(vm, f, indent=2, ensure_ascii=False)
            else:
                bad += 1

            # 매 epoch checkpoint
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
                    [epoch, f"{train_loss:.6f}", f"{vm['accuracy']:.6f}",
                     f"{score:.6f}", int(is_best)]
                )

            writer.add_scalar("Loss/train", train_loss, epoch)
            writer.add_scalar("F1/val_macro", score, epoch)
            print(
                f"[TCN {epoch:02d}] loss={train_loss:.5f} "
                f"val_macroF1={score:.4f}"
                + (" <- BEST" if is_best else "")
            )

            if bad >= PATIENCE:
                print(f"[TCN] early stop @ epoch {epoch}")
                break

    writer.close()

    if not best_path.exists():
        raise RuntimeError("TCN best.pt가 없습니다.")

    summary = {
        "stage": "tcn",
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
