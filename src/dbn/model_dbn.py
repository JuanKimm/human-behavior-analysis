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
        with np.load(path, allow_pickle=False) as d:
            required = {"alpha","cov_reg","var_floor","environment_balanced","z_mean","z_std","initial_prior","transition","means","covariances","class_counts","class_weight_mass","initial_counts_before_smoothing","transition_counts_before_smoothing","balance_info_json","features","class_names","inference"}
            missing = required.difference(d.files)
            if missing: raise ValueError(f"DBN checkpoint missing keys: {sorted(missing)}")
            obj = cls(alpha=float(d["alpha"]),cov_reg=float(d["cov_reg"]),var_floor=float(d["var_floor"]),
                      environment_balanced=bool(d["environment_balanced"]))
            if not np.isfinite([obj.alpha,obj.cov_reg,obj.var_floor]).all() or min(obj.alpha,obj.cov_reg,obj.var_floor)<=0:
                raise ValueError("DBN alpha/cov_reg/var_floor must be finite and positive")
            for key in ("z_mean","z_std","initial_prior","transition","means","covariances","class_counts","class_weight_mass","initial_counts_before_smoothing","transition_counts_before_smoothing"):
                if not np.isfinite(d[key]).all(): raise ValueError(f"DBN checkpoint {key} contains NaN/Inf")
            obj.z_mean=np.asarray(d["z_mean"],dtype=np.float64)
            obj.z_std=np.asarray(d["z_std"],dtype=np.float64)
            obj.initial_prior=np.asarray(d["initial_prior"],dtype=np.float64)
            obj.transition=np.asarray(d["transition"],dtype=np.float64)
            obj.means=np.asarray(d["means"],dtype=np.float64)
            obj.covariances=np.asarray(d["covariances"],dtype=np.float64)
            obj.class_counts=np.asarray(d["class_counts"])
            obj.class_weight_mass=np.asarray(d["class_weight_mass"])
            obj.initial_counts=np.asarray(d["initial_counts_before_smoothing"])
            obj.transition_counts=np.asarray(d["transition_counts_before_smoothing"])
            obj.balance_info=json.loads(str(d["balance_info_json"].item()))
            if tuple(d['features'].astype(str).tolist()) != cls.FEATURES: raise ValueError('DBN feature order is incompatible')
            class_names=tuple(d['class_names'].astype(str).tolist())
            if class_names not in (('Normal','Precursor','Danger'),('정상','전조','위험')): raise ValueError('DBN class order is incompatible')
            if str(d['inference'].item()) != 'filtering': raise ValueError('DBN checkpoint is not a filtering model')
        if obj.z_mean.shape!=(3,) or obj.z_std.shape!=(3,) or np.any(obj.z_std<=0): raise ValueError("DBN normalization arrays must have shape (3,) and positive std")
        if obj.initial_prior.shape!=(3,) or obj.transition.shape!=(3,3) or obj.means.shape!=(3,3) or obj.covariances.shape!=(3,3,3): raise ValueError("DBN class/feature arrays have invalid shapes")
        if any(a.shape!=(3,) for a in (obj.class_counts,obj.class_weight_mass,obj.initial_counts)) or obj.transition_counts.shape!=(3,3): raise ValueError("DBN count arrays have invalid shapes")
        if any(np.any(a<0) for a in (obj.class_counts,obj.class_weight_mass,obj.initial_counts,obj.transition_counts)): raise ValueError("DBN count arrays cannot be negative")
        if np.any(obj.initial_prior<0) or not np.isclose(obj.initial_prior.sum(),1,atol=1e-5): raise ValueError("DBN initial prior must be a probability vector")
        if np.any(obj.transition<0) or not np.allclose(obj.transition.sum(axis=1),1,atol=1e-5): raise ValueError("DBN transition rows must be probability vectors")
        for cov in obj.covariances:
            if not np.allclose(cov,cov.T,atol=1e-7): raise ValueError("DBN covariance is not symmetric")
            try: np.linalg.cholesky(cov)
            except np.linalg.LinAlgError as e: raise ValueError("DBN covariance is not positive definite") from e
        # var_floor is metadata from the training fit and is not applied during inference.
        return obj
