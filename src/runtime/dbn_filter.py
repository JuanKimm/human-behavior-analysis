from __future__ import annotations
import numpy as np
from dbn.model_dbn import FinalGaussianDBN, _logsumexp

class StatefulDBNFilter:
    """One-observation-at-a-time DBN Filtering for module/runtime use."""
    LABELS = ('Normal','Precursor','Danger')

    def __init__(self, checkpoint_path):
        self.model = FinalGaussianDBN.load(checkpoint_path)
        self.prev_log_post = None

    def reset(self):
        self.prev_log_post = None

    def update(self, t4: float, sp: float, sd: float):
        x = np.asarray([t4, sp, sd], dtype=np.float64)
        z = (x - self.model.z_mean) / self.model.z_std
        le = self.model._log_emission_one(z)
        log_prior = np.log(np.clip(self.model.initial_prior, 1e-300, None))
        log_trans = np.log(np.clip(self.model.transition, 1e-300, None))
        if self.prev_log_post is None:
            la = log_prior + le
        else:
            pred = _logsumexp(self.prev_log_post[:,None] + log_trans, axis=0)
            la = pred + le
        la = la - _logsumexp(la, axis=0)
        self.prev_log_post = la
        p = np.exp(la)
        idx = int(np.argmax(p))
        return {"label": self.LABELS[idx], "label_index": idx, "probabilities": p.tolist()}
