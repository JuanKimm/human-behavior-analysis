# -*- coding: utf-8 -*-
"""
Professor-plan evaluation metrics.

Saved for every test repeat:
- Accuracy
- Macro Precision
- Macro Recall
- Macro F1-score
- Balanced Accuracy
- per-class Precision / Recall / F1
- Confusion Matrix
- Danger Precision / Recall(Sensitivity) / FNR
- Precursor FPR (kept for the research objective)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

CLASS_NAMES = ["정상", "전조", "위험"]


def compute_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    labels = [0, 1, 2]

    cm = confusion_matrix(y_true, y_pred, labels=labels)

    precision = precision_score(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )
    recall = recall_score(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )
    f1 = f1_score(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )

    macro_precision = precision_score(
        y_true, y_pred, labels=labels, average="macro", zero_division=0
    )
    macro_recall = recall_score(
        y_true, y_pred, labels=labels, average="macro", zero_division=0
    )
    macro_f1 = f1_score(
        y_true, y_pred, labels=labels, average="macro", zero_division=0
    )

    # Precursor false-positive rate:
    # actual Normal or Danger predicted as Precursor.
    non_precursor = y_true != 1
    precursor_fp = int(np.sum(non_precursor & (y_pred == 1)))
    precursor_negative = int(np.sum(non_precursor))
    precursor_fpr = (
        precursor_fp / precursor_negative if precursor_negative else 0.0
    )

    danger_recall = float(recall[2])
    danger_fnr = float(1.0 - danger_recall)

    out = {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_true, y_pred)
        ),
        "per_class": {},
        "danger_precision": float(precision[2]),
        "danger_recall": danger_recall,
        "danger_sensitivity": danger_recall,
        "danger_fnr": danger_fnr,
        "precursor_fpr": float(precursor_fpr),
        "precursor_false_positives": precursor_fp,
        "precursor_negative_count": precursor_negative,
        "cm": cm.tolist(),
    }

    for i, name in enumerate(CLASS_NAMES):
        out["per_class"][name] = {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
        }

    return out


def save_metrics(metrics, prefix):
    prefix = Path(prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)

    with Path(str(prefix) + ".json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    with Path(str(prefix) + ".txt").open("w", encoding="utf-8") as f:
        f.write(f"n                  : {metrics['n']}\n")
        f.write(f"Accuracy           : {metrics['accuracy']:.6f}\n")
        f.write(f"Macro Precision    : {metrics['macro_precision']:.6f}\n")
        f.write(f"Macro Recall       : {metrics['macro_recall']:.6f}\n")
        f.write(f"Macro F1           : {metrics['macro_f1']:.6f}\n")
        f.write(f"Balanced Accuracy  : {metrics['balanced_accuracy']:.6f}\n")
        f.write(f"Danger Precision   : {metrics['danger_precision']:.6f}\n")
        f.write(f"Danger Recall      : {metrics['danger_recall']:.6f}\n")
        f.write(f"Danger FNR         : {metrics['danger_fnr']:.6f}\n")
        f.write(f"Precursor FPR      : {metrics['precursor_fpr']:.6f}\n\n")

        for name in CLASS_NAMES:
            m = metrics["per_class"][name]
            f.write(
                f"{name}: "
                f"P={m['precision']:.6f} "
                f"R={m['recall']:.6f} "
                f"F1={m['f1']:.6f}\n"
            )

        f.write("\nConfusion matrix:\n")
        f.write(str(np.asarray(metrics["cm"])))
        f.write("\n")
