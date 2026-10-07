# -*- coding: utf-8 -*-
"""
Frozen Skeleton-TCN + ST-DS-Transformer + CRF.

64-frame 입력에서:
- Short scale: frozen Skeleton-TCN이 16-frame x 4 chunks를 처리
- Mid scale: ST-DS-Transformer가 64-frame 전체를 처리
- 두 scale을 frame-aligned gated fusion
- 3-class pre-CRF emissions 생성
- CRF는 학습/최종 sequence decode에 사용

DBN에서는 CRF 결과를 쓰지 않고:
  softmax(pre-CRF emissions)
을 사용한다.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from tcn.model import SkeletonTCN


class LinearChainCRF(nn.Module):
    def __init__(self, num_tags):
        super().__init__()
        self.num_tags = int(num_tags)
        self.start_transitions = nn.Parameter(torch.empty(num_tags))
        self.end_transitions = nn.Parameter(torch.empty(num_tags))
        self.transitions = nn.Parameter(torch.empty(num_tags, num_tags))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.start_transitions, -0.1, 0.1)
        nn.init.uniform_(self.end_transitions, -0.1, 0.1)
        nn.init.uniform_(self.transitions, -0.1, 0.1)

    def _gold_score(self, emissions, tags):
        _, t, _ = emissions.shape
        score = self.start_transitions[tags[:, 0]]
        score = score + emissions[:, 0].gather(1, tags[:, 0:1]).squeeze(1)
        for i in range(1, t):
            prev_tag = tags[:, i - 1]
            cur_tag = tags[:, i]
            score = score + self.transitions[prev_tag, cur_tag]
            score = score + emissions[:, i].gather(1, cur_tag.unsqueeze(1)).squeeze(1)
        score = score + self.end_transitions[tags[:, -1]]
        return score

    def _log_partition(self, emissions):
        _, t, _ = emissions.shape
        alpha = self.start_transitions.unsqueeze(0) + emissions[:, 0]
        for i in range(1, t):
            scores = alpha.unsqueeze(2) + self.transitions.unsqueeze(0)
            alpha = torch.logsumexp(scores, dim=1) + emissions[:, i]
        return torch.logsumexp(
            alpha + self.end_transitions.unsqueeze(0), dim=1
        )

    def neg_log_likelihood(self, emissions, tags):
        gold = self._gold_score(emissions, tags)
        log_z = self._log_partition(emissions)
        b, t, _ = emissions.shape
        return (log_z - gold).sum() / float(b * t)

    @torch.no_grad()
    def decode(self, emissions):
        _, t, _ = emissions.shape
        score = self.start_transitions.unsqueeze(0) + emissions[:, 0]
        history = []

        for i in range(1, t):
            next_score = score.unsqueeze(2) + self.transitions.unsqueeze(0)
            best_score, best_prev = next_score.max(dim=1)
            score = best_score + emissions[:, i]
            history.append(best_prev)

        score = score + self.end_transitions.unsqueeze(0)
        best_last = score.argmax(dim=1)
        paths = [best_last]

        for best_prev in reversed(history):
            best_last = best_prev.gather(
                1, best_last.unsqueeze(1)
            ).squeeze(1)
            paths.append(best_last)

        paths.reverse()
        return torch.stack(paths, dim=1)


class MidScaleTransformer(nn.Module):
    def __init__(
        self,
        in_channels=3,
        d=96,
        heads=4,
        num_frames=64,
        num_joints=17,
        ffn_ratio=4,
    ):
        super().__init__()
        self.num_frames = int(num_frames)
        self.num_joints = int(num_joints)

        self.embed = nn.Linear(in_channels, d)
        self.spatial_pe = nn.Parameter(
            torch.zeros(1, 1, num_joints, d)
        )
        self.temporal_pe = nn.Parameter(
            torch.zeros(1, num_frames, 1, d)
        )
        nn.init.trunc_normal_(self.spatial_pe, std=0.02)
        nn.init.trunc_normal_(self.temporal_pe, std=0.02)

        self.spatial_attn = nn.MultiheadAttention(
            d, heads, batch_first=True
        )
        self.ln_s1 = nn.LayerNorm(d)
        self.spatial_ffn = nn.Sequential(
            nn.Linear(d, d * ffn_ratio),
            nn.GELU(),
            nn.Linear(d * ffn_ratio, d),
        )
        self.ln_s2 = nn.LayerNorm(d)

        self.temporal_attn = nn.MultiheadAttention(
            d, heads, batch_first=True
        )
        self.ln_t1 = nn.LayerNorm(d)
        self.temporal_ffn = nn.Sequential(
            nn.Linear(d, d * ffn_ratio),
            nn.GELU(),
            nn.Linear(d * ffn_ratio, d),
        )
        self.ln_t2 = nn.LayerNorm(d)

    def forward(self, x):
        b, t, v, _ = x.shape
        if t != self.num_frames or v != self.num_joints:
            raise ValueError(
                f"입력 shape 불일치: expected T={self.num_frames}, "
                f"V={self.num_joints}, actual T={t}, V={v}"
            )

        h = self.embed(x) + self.spatial_pe

        hs = h.reshape(b * t, v, -1)
        a, _ = self.spatial_attn(hs, hs, hs, need_weights=False)
        hs = self.ln_s1(hs + a)
        hs = self.ln_s2(hs + self.spatial_ffn(hs))
        h = hs.reshape(b, t, v, -1)

        h = h + self.temporal_pe
        ht = h.permute(0, 2, 1, 3).reshape(b * v, t, -1)
        a, _ = self.temporal_attn(ht, ht, ht, need_weights=False)
        ht = self.ln_t1(ht + a)
        ht = self.ln_t2(ht + self.temporal_ffn(ht))
        h = ht.reshape(b, v, t, -1).permute(0, 2, 1, 3)

        return h.mean(dim=2)  # (B,64,96)


class HybridSTDSTransformer(nn.Module):
    def __init__(
        self,
        tcn_ckpt_path,
        in_channels=3,
        d=96,
        heads=4,
        num_frames=64,
        num_joints=17,
        num_classes=3,
        ffn_ratio=4,
        use_crf=True,
        freeze_tcn=True,
    ):
        super().__init__()

        if int(num_frames) != 64:
            raise ValueError("현재 구조는 64-frame 입력만 지원합니다.")

        self.num_frames = int(num_frames)
        self.num_joints = int(num_joints)
        self.use_crf = bool(use_crf)
        self.freeze_tcn = bool(freeze_tcn)

        # Short scale
        self.tcn = SkeletonTCN(
            in_channels=in_channels,
            num_joints=num_joints,
            num_classes=2,
        )
        self._load_tcn_checkpoint(tcn_ckpt_path)

        if self.freeze_tcn:
            for p in self.tcn.parameters():
                p.requires_grad = False
            self.tcn.eval()

        # Mid scale
        self.encoder = MidScaleTransformer(
            in_channels=in_channels,
            d=d,
            heads=heads,
            num_frames=num_frames,
            num_joints=num_joints,
            ffn_ratio=ffn_ratio,
        )

        # Short/Mid fusion
        self.tcn_feat_proj = nn.Sequential(
            nn.Linear(256, d),
            nn.LayerNorm(d),
            nn.GELU(),
        )
        self.tcn_prob_proj = nn.Sequential(
            nn.Linear(2, d),
            nn.LayerNorm(d),
            nn.GELU(),
        )
        self.gate = nn.Sequential(
            nn.Linear(d * 2, d),
            nn.ReLU(inplace=True),
            nn.Linear(d, d),
            nn.Sigmoid(),
        )
        self.frame_head = nn.Sequential(
            nn.Linear(d, d),
            nn.ReLU(inplace=True),
            nn.Linear(d, num_classes),
        )

        self.crf = LinearChainCRF(num_classes) if use_crf else None

    def _load_tcn_checkpoint(self, ckpt_path):
        ckpt_path = Path(ckpt_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"TCN best.pt 없음: {ckpt_path}"
            )

        obj = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        state = (
            obj["model_state_dict"]
            if isinstance(obj, dict) and "model_state_dict" in obj
            else obj
        )
        self.tcn.load_state_dict(state, strict=True)

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_tcn:
            self.tcn.eval()
        return self

    def _tcn_chunks(self, x) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        """
        x: (B,64,17,C)
        TCN chunks:
          1~16 / 17~32 / 33~48 / 49~64
        """
        b, t, v, c = x.shape
        if t != 64:
            raise ValueError(f"TCN short scale requires T=64, actual={t}")

        chunks = (
            x.reshape(b, 4, 16, v, c)
             .reshape(b * 4, 16, v, c)
             .permute(0, 3, 1, 2)
             .contiguous()
        )

        if self.freeze_tcn:
            with torch.no_grad():
                feat = self.tcn.extract_features(chunks)
                logits = self.tcn.fc(feat)
        else:
            feat = self.tcn.extract_features(chunks)
            logits = self.tcn.fc(feat)

        chunk_feat = feat.reshape(b, 4, 256)
        chunk_logits = logits.reshape(b, 4, 2)
        chunk_probs = F.softmax(chunk_logits, dim=-1)

        # 각 16-frame chunk의 TCN 정보를 그 chunk의 16개 frame에 정렬
        frame_feat = chunk_feat.repeat_interleave(16, dim=1)
        frame_probs = chunk_probs.repeat_interleave(16, dim=1)

        return chunk_feat, chunk_probs, frame_feat, frame_probs

    def forward(self, x, return_aux=False):
        trans_feat = self.encoder(x)

        (
            chunk_feat,
            chunk_probs,
            tcn_frame_feat_raw,
            tcn_frame_probs,
        ) = self._tcn_chunks(x)

        tcn_feat = (
            self.tcn_feat_proj(tcn_frame_feat_raw)
            + self.tcn_prob_proj(tcn_frame_probs)
        )

        gate = self.gate(
            torch.cat([trans_feat, tcn_feat], dim=-1)
        )
        fused = gate * trans_feat + (1.0 - gate) * tcn_feat

        # CRF 적용 전 3-class logits
        emissions = self.frame_head(fused)

        if not return_aux:
            return emissions

        aux = {
            "transformer_features": trans_feat,
            "tcn_frame_features": tcn_feat,
            "tcn_chunk_features": chunk_feat,
            "tcn_chunk_probs": chunk_probs,
            "tcn_frame_probs": tcn_frame_probs,
            "gate": gate,
        }
        return emissions, aux

    def get_pre_crf_outputs(self, x):
        """
        DBN용.
        CRF를 거치지 않은 ST-DS 3-class probability와
        4개 TCN chunk probability를 함께 반환한다.

        DBN에서는 tcn_chunk_probs[:, 3]을 T4(49~64 frame)로 사용하면 된다.
        """
        emissions, aux = self.forward(x, return_aux=True)
        return {
            "emissions": emissions,
            "stds_precrf_probs": F.softmax(emissions, dim=-1),
            "tcn_chunk_probs": aux["tcn_chunk_probs"],
        }

    def decode(self, x=None, emissions=None):
        if emissions is None:
            if x is None:
                raise ValueError("x 또는 emissions 중 하나가 필요합니다.")
            emissions = self.forward(x)

        if self.crf is None:
            return emissions.argmax(dim=-1)
        return self.crf.decode(emissions)

    def crf_nll(self, emissions, tags):
        if self.crf is None:
            raise RuntimeError("CRF가 비활성화되어 있습니다.")
        return self.crf.neg_log_likelihood(emissions, tags)
