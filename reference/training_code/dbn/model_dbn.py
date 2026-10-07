# -*- coding: utf-8 -*-
"""
최종 DBN:
  observation = [T4, S_P, S_D]

T4:
  Skeleton-TCN의 49~64 frame chunk
  log[P(Danger) / P(NonDanger)]

S_P:
  CRF 적용 전 ST-DS 마지막(64번째) frame probability
  log[P(Precursor) / P(Normal)]

S_D:
  CRF 적용 전 ST-DS 마지막(64번째) frame probability
  log[P(Danger) / P(Normal)]
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

NUM_CLASSES = 3
CLASS_NAMES = ["정상", "전조", "위험"]


def _logsumexp(a, axis=None):
    a = np.asarray(a, dtype=np.float64)
    m = np.max(a, axis=axis, keepdims=True)
    out = m + np.log(
        np.sum(np.exp(a - m), axis=axis, keepdims=True)
    )
    if axis is not None:
        out = np.squeeze(out, axis=axis)
    return out


def environment_weights(rows):
    env_window_counts = Counter(
        str(r["env_name"]) for r in rows
    )

    video_to_env = {}
    for r in rows:
        video_to_env[int(r["video_idx"])] = str(r["env_name"])
    env_video_counts = Counter(video_to_env.values())

    emission_w = np.asarray(
        [
            1.0 / env_window_counts[str(r["env_name"])]
            for r in rows
        ],
        dtype=np.float64,
    )
    sequence_w = np.asarray(
        [
            1.0 / env_video_counts[str(r["env_name"])]
            for r in rows
        ],
        dtype=np.float64,
    )

    emission_w /= emission_w.mean()
    sequence_w /= sequence_w.mean()

    info = {
        "env_window_counts": dict(
            sorted(env_window_counts.items())
        ),
        "env_video_counts": dict(
            sorted(env_video_counts.items())
        ),
        "emission_weight_rule": (
            "inverse environment window count, normalized to mean 1"
        ),
        "sequence_weight_rule": (
            "inverse environment video count, normalized to mean 1"
        ),
    }
    return emission_w, sequence_w, info


class FinalGaussianDBN:
    FEATURES = ("T4", "S_P", "S_D")

    def __init__(
        self,
        alpha=0.1,
        cov_reg=1e-3,
        var_floor=1e-6,
        environment_balanced=True,
    ):
        self.alpha = float(alpha)
        self.cov_reg = float(cov_reg)
        self.var_floor = float(var_floor)
        self.environment_balanced = bool(environment_balanced)

        self.z_mean = None
        self.z_std = None
        self.initial_prior = None
        self.transition = None
        self.means = None
        self.covariances = None
        self.class_counts = None
        self.class_weight_mass = None
        self.initial_counts = None
        self.transition_counts = None
        self.balance_info = None

    @staticmethod
    def _arrays(rows):
        x = np.asarray(
            [[r[k] for k in FinalGaussianDBN.FEATURES] for r in rows],
            dtype=np.float64,
        )
        y = np.asarray(
            [r["target_label"] for r in rows],
            dtype=np.int64,
        )
        vids = np.asarray(
            [r["video_idx"] for r in rows],
            dtype=np.int64,
        )
        starts = np.asarray(
            [r["window_abs_start"] for r in rows],
            dtype=np.int64,
        )
        return x, y, vids, starts

    @staticmethod
    def _group_indices(vids, starts):
        groups = defaultdict(list)
        for i, vi in enumerate(vids.tolist()):
            groups[int(vi)].append(i)
        for vi in groups:
            groups[vi].sort(
                key=lambda idx: int(starts[idx])
            )
        return dict(groups)

    def fit(self, rows):
        x, y, vids, starts = self._arrays(rows)
        if len(x) == 0:
            raise RuntimeError("DBN 학습 row가 0개입니다.")

        self.z_mean = x.mean(axis=0)
        self.z_std = x.std(axis=0, ddof=0)
        self.z_std = np.where(
            self.z_std < 1e-8, 1.0, self.z_std
        )
        z = (x - self.z_mean) / self.z_std

        if self.environment_balanced:
            emission_w, sequence_w, info = environment_weights(rows)
            self.balance_info = info
        else:
            emission_w = np.ones(len(rows), dtype=np.float64)
            sequence_w = np.ones(len(rows), dtype=np.float64)
            self.balance_info = {
                "emission_weight_rule": "uniform",
                "sequence_weight_rule": "uniform",
            }

        d = len(self.FEATURES)
        self.means = np.zeros(
            (NUM_CLASSES, d), dtype=np.float64
        )
        self.covariances = np.zeros(
            (NUM_CLASSES, d, d), dtype=np.float64
        )
        self.class_counts = np.bincount(
            y, minlength=NUM_CLASSES
        ).astype(np.int64)
        self.class_weight_mass = np.zeros(
            NUM_CLASSES, dtype=np.float64
        )

        for c in range(NUM_CLASSES):
            mask = y == c
            xc = z[mask]
            wc = emission_w[mask]

            if len(xc) == 0:
                raise RuntimeError(
                    f"DBN 학습 데이터에 class={c}가 없습니다."
                )

            sw = float(wc.sum())
            self.class_weight_mass[c] = sw

            mu = (wc[:, None] * xc).sum(axis=0) / sw
            self.means[c] = mu

            centered = xc - mu
            denom = sw - float(np.dot(wc, wc)) / sw
            if denom <= 1e-12:
                denom = sw

            cov = (
                (centered * wc[:, None]).T @ centered
                / denom
            )
            cov = np.atleast_2d(
                np.asarray(cov, dtype=np.float64)
            )
            cov += self.cov_reg * np.eye(
                d, dtype=np.float64
            )
            self.covariances[c] = cov

        groups = self._group_indices(vids, starts)
        init_counts = np.zeros(
            NUM_CLASSES, dtype=np.float64
        )
        trans_counts = np.zeros(
            (NUM_CLASSES, NUM_CLASSES), dtype=np.float64
        )

        for idxs in groups.values():
            if not idxs:
                continue

            first = idxs[0]
            init_counts[y[first]] += sequence_w[first]

            for a, b in zip(idxs[:-1], idxs[1:]):
                trans_counts[
                    y[a], y[b]
                ] += sequence_w[a]

        self.initial_counts = init_counts.copy()
        self.transition_counts = trans_counts.copy()

        self.initial_prior = (
            init_counts + self.alpha
        ) / (
            init_counts.sum()
            + self.alpha * NUM_CLASSES
        )

        trans_sm = trans_counts + self.alpha
        self.transition = (
            trans_sm
            / trans_sm.sum(axis=1, keepdims=True)
        )
        return self

    def transform(self, rows):
        x = np.asarray(
            [[r[k] for k in self.FEATURES] for r in rows],
            dtype=np.float64,
        )
        return (x - self.z_mean) / self.z_std

    def _log_emission_one(self, zrow):
        d = len(self.FEATURES)
        out = np.zeros(
            NUM_CLASSES, dtype=np.float64
        )
        const = d * np.log(2.0 * np.pi)

        for c in range(NUM_CLASSES):
            diff = zrow - self.means[c]
            cov = self.covariances[c]

            sign, logdet = np.linalg.slogdet(cov)
            if sign <= 0:
                cov = (
                    cov
                    + max(self.cov_reg, 1e-6)
                    * 10.0
                    * np.eye(d)
                )
                sign, logdet = np.linalg.slogdet(cov)

            try:
                sol = np.linalg.solve(cov, diff)
            except np.linalg.LinAlgError:
                sol = np.linalg.pinv(cov) @ diff

            out[c] = -0.5 * (
                const + logdet + diff @ sol
            )
        return out

    def filter(self, rows):
        z = self.transform(rows)
        _, _, vids, starts = self._arrays(rows)
        groups = self._group_indices(vids, starts)

        post = np.zeros(
            (len(rows), NUM_CLASSES),
            dtype=np.float64,
        )

        log_prior = np.log(
            np.clip(self.initial_prior, 1e-300, None)
        )
        log_trans = np.log(
            np.clip(self.transition, 1e-300, None)
        )

        for idxs in groups.values():
            prev_log_post = None

            for pos, idx in enumerate(idxs):
                le = self._log_emission_one(z[idx])

                if pos == 0:
                    la = log_prior + le
                else:
                    pred = _logsumexp(
                        prev_log_post[:, None] + log_trans,
                        axis=0,
                    )
                    la = pred + le

                la = la - _logsumexp(la, axis=0)
                post[idx] = np.exp(la)
                prev_log_post = la

        return post

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            model_name=np.asarray(
                "Final T4 + pre-CRF ST-DS environment-balanced DBN"
            ),
            features=np.asarray(self.FEATURES),
            environment_balanced=np.asarray(
                self.environment_balanced
            ),
            alpha=np.asarray(self.alpha),
            cov_reg=np.asarray(self.cov_reg),
            var_floor=np.asarray(self.var_floor),
            z_mean=self.z_mean,
            z_std=self.z_std,
            initial_prior=self.initial_prior,
            transition=self.transition,
            means=self.means,
            covariances=self.covariances,
            class_counts=self.class_counts,
            class_weight_mass=self.class_weight_mass,
            initial_counts_before_smoothing=self.initial_counts,
            transition_counts_before_smoothing=self.transition_counts,
            balance_info_json=np.asarray(
                json.dumps(
                    self.balance_info,
                    ensure_ascii=False,
                )
            ),
            class_names=np.asarray(CLASS_NAMES),
            tcn_source=np.asarray(
                "chunk4 = frames 49-64"
            ),
            stds_probability_source=np.asarray(
                "pre-CRF softmax at frame 64"
            ),
            inference=np.asarray("filtering"),
        )

    @classmethod
    def load(cls, path):
        d = np.load(path, allow_pickle=True)

        obj = cls(
            alpha=float(d["alpha"]),
            cov_reg=float(d["cov_reg"]),
            var_floor=float(d["var_floor"]),
            environment_balanced=bool(
                d["environment_balanced"]
            ),
        )
        obj.z_mean = d["z_mean"]
        obj.z_std = d["z_std"]
        obj.initial_prior = d["initial_prior"]
        obj.transition = d["transition"]
        obj.means = d["means"]
        obj.covariances = d["covariances"]
        obj.class_counts = d["class_counts"]
        obj.class_weight_mass = d["class_weight_mass"]
        obj.initial_counts = d[
            "initial_counts_before_smoothing"
        ]
        obj.transition_counts = d[
            "transition_counts_before_smoothing"
        ]
        obj.balance_info = json.loads(
            str(d["balance_info_json"])
        )
        return obj
