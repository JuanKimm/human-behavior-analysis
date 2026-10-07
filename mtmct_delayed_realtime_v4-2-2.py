"""v4-2-2: v4-2 기반, 카메라 간 기존 ID 보호와 위치 기준 복구.

실행: python mtmc_delayed_realtime_v4-2-2.py --show-preview --pace-input
검사: python mtmc_delayed_realtime_v4-2-2.py --self-test
기본 입력: videos/color6-415.mp4, videos/color6-435.mp4
기본 출력: MTMCT_delayed_results_v4-2-2/run_.../

변경 사항:
- 양 카메라 관측으로 배정 전 기존 소유자를 확인해 처리 순서에 따른 ID 탈취 방지.
- 모호하지 않은 카메라 쌍 + 안정적인 기존 소유자 + 여러 관측의 위치/이동 일치가
  있을 때만 해당 카메라의 잘못된 위치 기준 복구. 미래 관측은 증거로만 사용하며
  위치 갱신에는 현재 source frame만 사용. 외형만으로 복구하지 않음.
- 위치 기준 갱신(.45)과 충돌 검사(.20 이상)를 분리. bbox 크기 변화/가림이면
  연결된 최근 정상 위치를 짧게 사용하고 시간 경과만큼 충돌 허용 오차를 늘림.
- 기존 YOLO/OSNet/BoT-SORT/캘리브레이션, 매칭 비용, 0.3초 지연, 신규 확인,
  로컬 궤적 복구, 마지막 확정 매칭 후 최소 300초 보관 정책 유지.

추가 Config: cross_owner_seconds(.3), cross_owner_similarity(.75),
conflict_history_seconds(.3). 값들은 실제 영상 재검증이 필요한 초기값.
추가 진단: current_owner_position_conflict, cross_view_recovery_supported,
cross_view_anchor_reset. 이전 출력/ID 소급 변경 없음. 인원수 고정/정답 주입 없음.
패키지/가중치: 기존 v4-2 가상환경 그대로 사용.
합성 회귀 검사는 실제 영상 정확도 개선을 보장하지 않음.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import inspect
import json
import math
import os
import sys
import time
import urllib.request
from collections import Counter, deque
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np


VERSION = "20261001-delayed-bbox-v4-2-2-cross-owner"
BASELINE_SOURCE_SHA256 = "f7ec2d593851f1cc9e07ae9032a79b749696ff2a0db178a327dba41f3c8839bf"
VIDEO1_PATH = "videos/color7-415.mp4"
VIDEO2_PATH = "videos/color7-435.mp4"
REID_WEIGHTS_PATH = "weights/osnet_x1_0_msmt17.pth"
REID_SHA256 = "48df972f72887b95cf3b43b3a07c3a7d2398381aea0f9cae64a7ef11d512b727"
REID_URL = (
    "https://huggingface.co/kaiyangzhou/osnet/resolve/"
    "a5c5cc037c24235cda3b21085b93ad77c9616224/"
    "osnet_x1_0_msmt17_combineall_256x128_amsgrad_ep150_stp60_lr0.0015_"
    "b64_fb10_softmax_labelsmooth_flip_jitter.pth"
)
CAM1_IMAGE_POINTS = np.array([[353, 389], [115, 623], [1127, 621], [1217, 392]], np.float32)
CAM2_IMAGE_POINTS = np.array([[15, 485], [332, 711], [1231, 593], [906, 422]], np.float32)
WORLD_POINTS_CAM1 = np.array([[0., 0.], [0., 3.06], [4.08, 3.06], [6.12, 0.]], np.float32)
WORLD_POINTS_CAM2 = np.array([[1.02, 0.], [3.06, 3.06], [7.14, 3.06], [7.14, 0.]], np.float32)
CALIBRATION_IMAGE_SIZES = {1: (1280, 720), 2: (1280, 720)}
TEMPORAL_CORRECTIONS = {
    ("color2-415.mp4", "color2-435.mp4"): ("none",),
    ("color6-415.mp4", "color6-435.mp4"): ("drift", -0.005045, -1.380698),
    ("color7-415.mp4", "color7-435.mp4"): ("step", 150, -1),
}


def unit(value):
    if value is None:
        return None
    a = np.asarray(value, np.float32).reshape(-1)
    norm = float(np.linalg.norm(a))
    return a / norm if np.isfinite(a).all() and norm > 1e-8 else None


def cosine(a, b):
    return None if a is None or b is None else float(np.clip(np.dot(a, b), -1., 1.))


def iou(a, b):
    intersection = float(np.prod(np.maximum(0., np.minimum(a[2:], b[2:]) - np.maximum(a[:2], b[:2]))))
    return intersection / max(1., float(np.prod(a[2:] - a[:2]) + np.prod(b[2:] - b[:2])) - intersection)


def assign(cost, limit):
    """LAPJV with explicit unmatched option, never force an invalid pair."""
    import lap
    cost = np.asarray(cost, np.float64)
    if cost.size == 0:
        return []
    safe = np.where(np.isfinite(cost) & (cost <= limit), cost, 1e6)
    _, rows, _ = lap.lapjv(safe, extend_cost=True, cost_limit=float(limit))
    return [(i, int(j)) for i, j in enumerate(rows) if j >= 0 and safe[i, j] <= limit]


def recovery_assignment(cost, limit, margin, established):
    """For failed-pair fallback, compare COMPLETE assignments, including abstention.

    Per-row nearest-neighbor margins cannot represent competition: A may use
    either ID while B has only one. Exclude each chosen edge and solve again to
    measure ambiguity of the actual one-to-one solution. LAPJV's objective is
    sum(cost-limit) for matched edges plus an irrelevant constant.
    """
    cost = np.asarray(cost, float)
    best = assign(cost, limit)
    objective = sum(cost[i, j] - limit for i, j in best)
    accepted = []
    for i, j in best:
        if (i, j) in established:
            accepted.append((i, j))
            continue
        excluded = cost.copy()
        excluded[i, j] = math.inf
        alternative = assign(excluded, limit)
        regret = sum(cost[r, c] - limit for r, c in alternative) - objective
        if regret >= margin - 1e-8:
            accepted.append((i, j))
    return accepted


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


class Timings:
    def __init__(self):
        self.total, self.calls = Counter(), Counter()

    @contextmanager
    def measure(self, name):
        start = time.perf_counter()
        try:
            yield
        finally:
            self.total[name] += time.perf_counter() - start
            self.calls[name] += 1

    def report(self):
        return {k: {"seconds": v, "calls": self.calls[k], "mean_ms": 1000 * v / self.calls[k]}
                for k, v in self.total.items()}


class Calibration:
    def __init__(self):
        import cv2
        self.h = {}
        self.report = {"unit": "m", "anchor": "bbox_bottom_center", "fit": {},
                       "independent_validation": "NOT_MEASURED", "matching_roi_gate": False}
        for cam, src, dst in ((1, CAM1_IMAGE_POINTS, WORLD_POINTS_CAM1),
                              (2, CAM2_IMAGE_POINTS, WORLD_POINTS_CAM2)):
            h, _ = cv2.findHomography(src.astype(np.float64), dst.astype(np.float64), method=0)
            if h is None or not np.isfinite(h).all() or abs(h[2, 2]) < 1e-12:
                raise ValueError(f"CAM{cam}: invalid calibration")
            self.h[cam] = h / h[2, 2]
            fitted = cv2.perspectiveTransform(src.astype(np.float64).reshape(-1, 1, 2), self.h[cam]).reshape(-1, 2)
            self.report["fit"][cam] = {"image_points": src.tolist(), "world_points": dst.tolist(),
                "H_image_to_world": self.h[cam].tolist(),
                "fit_rmse_m": float(np.sqrt(np.mean(np.sum((fitted - dst) ** 2, axis=1))))}
        h21 = np.linalg.inv(self.h[1]) @ self.h[2]
        self.report["H_camera2_to_camera1"] = (h21 / h21[2, 2]).tolist()

    def check_size(self, cam, size, allow_resize):
        if tuple(size) != CALIBRATION_IMAGE_SIZES[cam] and not allow_resize:
            raise ValueError(f"CAM{cam} 영상 {size}, 기준 {CALIBRATION_IMAGE_SIZES[cam]}. "
                             "동일 화각의 단순 resize인 경우만 --allow-resize-calibration 사용")

    def project(self, cam, point, size):
        p = np.asarray(point, np.float64) * np.asarray(CALIBRATION_IMAGE_SIZES[cam]) / np.asarray(size)
        q = self.h[cam] @ np.r_[p, 1.]
        if not np.isfinite(q).all() or abs(q[2]) < 1e-8:
            return None
        world = q[:2] / q[2]
        return world if np.linalg.norm(world) < 1000 else None


@dataclass
class Config:
    delay_seconds: float = .3
    cross_distance: float = 1.2
    sync_tolerance: float = .08
    cross_similarity: float = .48
    pair_cost_limit: float = .58
    identity_cost_limit: float = .64
    ambiguity_margin: float = .04
    min_evidence: int = 3
    min_evidence_seconds: float = .06
    max_speed: float = 3.
    max_gap: float = 3.
    feature_quality: float = .25
    position_quality: float = .20
    # Pixel thresholds are specified for 1280x720 and scaled with input size.
    min_bbox_width: int = 24
    min_bbox_height: int = 80
    birth_min_width: int = 32
    birth_min_height: int = 120
    new_track_confidence: float = .50
    birth_min_hits: int = 3
    birth_min_seconds: float = .10
    birth_window_seconds: float = .50
    reid_interval: int = 2
    local_buffer_seconds: float = 3.
    occlusion_overlap: float = .35
    partial_height_ratio: float = .78
    duplicate_iou: float = .85
    identity_memory_seconds: float = 300.
    feature_history_seconds: float = 1.
    severe_overlap: float = .75
    dormant_similarity: float = .75
    # Soft cost preference, not a time/hit gate. Invalid old matches get no bonus.
    continuity_margin: float = .04
    batch_pair_fallback: bool = True
    # Stronger requirements apply to storing a motion anchor, not detecting people.
    position_anchor_quality: float = .45
    motion_recovery_similarity: float = .85
    motion_recovery_samples: int = 3
    motion_recovery_seconds: float = .06
    motion_recovery_speed: float = 8.0  # Only verified existing-ID recovery, not new matches.
    motion_history_seconds: float = .60
    motion_sample_gap: float = .15
    cross_owner_seconds: float = .30
    cross_owner_similarity: float = .75
    conflict_history_seconds: float = .30

    def __post_init__(self):
        if not np.isfinite(self.identity_memory_seconds) or self.identity_memory_seconds < 300.:
            raise ValueError("identity_memory_seconds는 최소 300초여야 합니다.")


@dataclass
class Observation:
    cam: int
    frame: int
    timestamp: float
    bbox: np.ndarray | None  # Actual detector output; absent boxes are never displayed.
    confidence: float
    quality: float
    position_quality: float
    world: np.ndarray | None
    feature: np.ndarray | None = None
    color: np.ndarray | None = None
    local_id: int = -1
    epoch: int = 0
    occluded: bool = False
    partial: bool = False
    tracking_bbox: np.ndarray | None = None
    association_bbox: np.ndarray | None = None
    hint_local_id: int = -1
    continuity_ok: bool = False
    reliable_age: float = math.inf
    image_size: tuple = (1280, 720)
    local_confirmed: bool = True
    association_confidence: float | None = None
    overlap: float = 0.
    appearance_update_ok: bool = True
    position_usable: bool | None = None
    track_feature: np.ndarray | None = None
    track_feature_time: float | None = None
    decision: str = ""
    match_details: dict = field(default_factory=dict)
    position_anchor_ok: bool | None = None
    position_anchor_reason: str = "unassessed"

    @property
    def key(self):
        return self.cam, self.local_id, self.epoch


@dataclass
class Packet:
    sequence: int
    timestamp: float
    frames: dict
    indices: dict
    times: dict
    fresh: dict
    observations: dict = field(default_factory=dict)
    arrival: float = 0.
    diagnostics: dict = field(default_factory=dict)


def correction_lag(index1, correction):
    if correction[0] == "none":
        return 0
    if correction[0] == "step":
        return int(correction[2]) if index1 >= correction[1] else 0
    if correction[0] == "drift":
        return int(round(correction[1] * index1 + correction[2]))
    raise ValueError(f"Unknown correction: {correction}")


class VideoPairReader:
    """Sequential decoding, common sampling FPS, fail on premature decode failure."""
    def __init__(self, paths, start_frame, end_frame, correction, offset=0.):
        import cv2
        self.caps, self.info, self.last_images, self.last_times = {}, {}, {}, {}
        self.indices = {1: -1, 2: -1}
        self.sequence = 0
        self.reason = None
        self.correction, self.offset = correction, offset
        self.reused, self.skipped = Counter(), Counter()
        try:
            for cam, path in enumerate(paths, 1):
                if not Path(path).is_file():
                    raise FileNotFoundError(f"영상 파일 없음: {path}")
                cap = cv2.VideoCapture(str(path))
                self.caps[cam] = cap
                if not cap.isOpened():
                    raise RuntimeError(f"영상 열기 실패: {path}")
                meta = {"width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                        "fps": float(cap.get(cv2.CAP_PROP_FPS)),
                        "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT))}
                if not all(np.isfinite(v) and v > 0 for v in meta.values()):
                    raise ValueError(f"CAM{cam} invalid metadata: {meta}")
                self.info[cam] = meta
            self.fps = min(x["fps"] for x in self.info.values())
            self.start = start_frame / self.info[1]["fps"]
            self.end1 = min(self.info[1]["frames"] - 1,
                            end_frame if end_frame is not None else self.info[1]["frames"] - 1)
            if start_frame < 0 or start_frame > self.end1:
                raise ValueError("start-frame/end-frame 범위 오류")
            indices, _ = self.targets(0)
            if indices[2] < 0 or indices[2] >= self.info[2]["frames"]:
                raise ValueError("CAM2 시작 대응 프레임 없음. offset 또는 start-frame 조정 필요")
            for cam, idx in indices.items():
                if idx and not self.caps[cam].set(cv2.CAP_PROP_POS_FRAMES, idx):
                    raise RuntimeError(f"CAM{cam} 시작 seek 실패")
                self.indices[cam] = idx - 1
        except BaseException:
            self.close()
            raise

    def targets(self, sequence):
        t = self.start + sequence / self.fps
        f1, f2 = self.info[1]["fps"], self.info[2]["fps"]
        i1 = int(round(t * f1))
        lag = correction_lag(i1, self.correction)
        i2 = int(round(i1 * f2 / f1 + self.offset * f2)) + lag
        return {1: i1, 2: i2}, {1: i1 / f1, 2: (i2 - lag) / f2 - self.offset}

    def next(self):
        target, times = self.targets(self.sequence)
        if target[1] > self.end1:
            self.reason = "requested_end_or_cam1_eof"
            return None
        if target[2] >= self.info[2]["frames"]:
            self.reason = "cam2_eof_common_interval_end"
            return None
        fresh = {}
        for cam in (1, 2):
            old = self.indices[cam]
            if target[cam] < old or target[cam] < 0:
                raise ValueError("시간 보정이 프레임을 역행시킵니다. 보정 설정을 확인하세요.")
            while self.indices[cam] < target[cam]:
                ok, frame = self.caps[cam].read()
                if not ok:
                    raise RuntimeError(f"CAM{cam}: 메타데이터상 EOF 이전 decode 실패 "
                                       f"(frame {self.indices[cam] + 1})")
                self.indices[cam] += 1
                self.last_images[cam] = frame
            fresh[cam] = old != self.indices[cam]
            if fresh[cam]:
                self.skipped[cam] += max(0, target[cam] - old - 1)
                self.last_times[cam] = times[cam]
            else:
                self.reused[cam] += 1
        packet = Packet(self.sequence, self.start + self.sequence / self.fps,
                        dict(self.last_images), dict(self.indices), dict(self.last_times), fresh,
                        arrival=time.perf_counter())
        self.sequence += 1
        return packet

    def close(self):
        for cap in self.caps.values():
            cap.release()


def sha256_file(path):
    value = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def prepare_weights(path, offline, expected_hash):
    target = Path(path)
    official = target.resolve() == Path(REID_WEIGHTS_PATH).resolve()
    expected = expected_hash or (REID_SHA256 if official else None)
    if not target.is_file():
        if offline or not official:
            raise FileNotFoundError(f"OSNet 가중치 없음: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        temp = target.with_name(target.name + f".{os.getpid()}.part")
        try:
            print("[준비] OSNet MSMT17 가중치 다운로드", flush=True)
            with urllib.request.urlopen(REID_URL, timeout=60) as source, open(temp, "wb") as dest:
                for block in iter(lambda: source.read(1024 * 1024), b""):
                    dest.write(block)
            if sha256_file(temp) != REID_SHA256:
                raise RuntimeError("OSNet 다운로드 SHA256 불일치")
            os.replace(temp, target)
        finally:
            temp.unlink(missing_ok=True)
    actual = sha256_file(target)
    if expected and actual.lower() != expected.lower():
        raise RuntimeError("OSNet SHA256 불일치. 사용자 학습 가중치는 별도 경로로 지정하세요.")
    return target, actual


class FeatureEncoder:
    def __init__(self, path, device, torch_module, half=False):
        import torchreid
        self.torch, self.device, self.half = torch_module, device, half
        if not hasattr(torchreid, "models"):
            raise RuntimeError("KaiyangZhou/deep-person-reid의 torchreid가 필요합니다.")
        self.model = torchreid.models.build_model(name="osnet_x1_0", num_classes=1000, pretrained=False)
        checkpoint = self.torch.load(str(path), map_location="cpu", weights_only=True)
        raw = checkpoint.get("state_dict", checkpoint)
        raw = {k.removeprefix("module."): v for k, v in raw.items()}
        dst = self.model.state_dict()
        matched = {k: v for k, v in raw.items() if k in dst and not k.startswith("classifier.")
                   and hasattr(v, "shape") and v.shape == dst[k].shape}
        missing = [k for k in dst if not k.startswith("classifier.") and
                   not k.endswith("num_batches_tracked") and k not in matched]
        if missing:
            raise ValueError(f"OSNet backbone 가중치 누락: {missing[:8]}")
        self.model.load_state_dict(matched, strict=False)
        self.model.eval().to(device)
        if half:
            self.model.half()
        self.mean = np.array([.485, .456, .406], np.float32)
        self.std = np.array([.229, .224, .225], np.float32)

    def encode(self, crops):
        import cv2
        output = []
        with self.torch.inference_mode():
            for start in range(0, len(crops), 32):
                images = []
                for crop in crops[start:start + 32]:
                    rgb = cv2.cvtColor(cv2.resize(crop, (128, 256)), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.
                    images.append(((rgb - self.mean) / self.std).transpose(2, 0, 1))
                batch = self.torch.from_numpy(np.ascontiguousarray(np.stack(images))).to(self.device)
                values = self.model(batch.half() if self.half else batch)
                if isinstance(values, (tuple, list)):
                    values = values[0]
                output.extend(self.torch.nn.functional.normalize(values.float(), dim=1).cpu().numpy())
        return output


class FeatureBoxes:
    """Keep original detection indices during upstream Boolean/index slicing."""
    def __init__(self, detections, indices=None):
        self.items = list(detections)
        self.indices = np.arange(len(self.items)) if indices is None else np.asarray(indices)
        self.xyxy = np.asarray([d.association_bbox if d.association_bbox is not None else d.bbox
                                for d in self.items], np.float32).reshape(-1, 4)
        self.xywh = self.xyxy.copy()
        self.xywh[:, :2] = (self.xyxy[:, :2] + self.xyxy[:, 2:]) / 2
        self.xywh[:, 2:] -= self.xyxy[:, :2]
        self.conf = np.array([d.confidence if d.association_confidence is None else d.association_confidence
                              for d in self.items], np.float32)
        self.cls = np.zeros(len(self.items), np.float32)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, mask):
        ids = np.atleast_1d(np.arange(len(self))[mask])
        return FeatureBoxes([self.items[i] for i in ids], self.indices[ids])


def make_local_tracker(fps, cfg):
    from ultralytics.trackers.bot_sort import BOTSORT, BOTrack
    if "results" not in inspect.signature(BOTSORT.init_track).parameters:
        raise RuntimeError("지원하지 않는 Ultralytics tracker API: init_track(results, img) 필요")

    class PersonTrack(BOTrack):
        def __init__(self, xywh, obs, source_idx):
            self.incoming_quality = obs.quality
            self.source_idx = int(source_idx)
            self.observation = obs
            self.reliable_box = None
            self.reliable_time = None
            self.full_box = None if obs.partial else obs.bbox.copy()
            self.full_box_time = obs.timestamp
            self.box_velocity = np.zeros(4, np.float32)
            self.last_observation_time = obs.timestamp
            self.feature_time = None
            if not obs.partial and obs.position_quality >= cfg.position_quality:
                self.reliable_box, self.reliable_time = obs.bbox.copy(), obs.timestamp
            score = obs.confidence if obs.association_confidence is None else obs.association_confidence
            super().__init__(xywh, score, 0, None if obs.partial else obs.feature)

        def update_features(self, feat):
            if self.observation.partial:
                return
            f = unit(feat)
            if f is None:
                return
            self.curr_feat = f
            old = getattr(self, "smooth_feat", None)
            if old is None:
                self.smooth_feat = f
                self.feature_time = self.observation.timestamp
            elif self.observation.appearance_update_ok and self.incoming_quality >= .30:
                alpha = .98 - .13 * self.incoming_quality
                self.smooth_feat = unit(alpha * old + (1 - alpha) * f)
                self.feature_time = self.observation.timestamp

        def accept_observation(self, new_track):
            self.incoming_quality = new_track.incoming_quality
            self.source_idx = new_track.source_idx
            self.observation = new_track.observation
            obs = self.observation
            self.last_observation_time = obs.timestamp
            # A full-size overlapping box can still constrain image motion,
            # while its appearance/floor coordinate remain untrusted. Otherwise
            # stale pre-overlap velocity can drive two predictions past each other.
            if not obs.partial:
                if self.full_box is not None and obs.timestamp > self.full_box_time:
                    dt = obs.timestamp - self.full_box_time
                    velocity = (obs.bbox - self.full_box) / dt
                    height = max(20., self.full_box[3] - self.full_box[1])
                    if np.max(np.abs(velocity)) < 3 * height:
                        self.box_velocity = .8 * self.box_velocity + .2 * velocity
                self.full_box, self.full_box_time = obs.bbox.copy(), obs.timestamp
            if not obs.partial and obs.position_quality >= cfg.position_quality:
                self.reliable_box, self.reliable_time = obs.bbox.copy(), obs.timestamp

        def update(self, new_track, frame_id):
            # A partial crop is not a new full-body Kalman measurement. Preserve
            # the predicted state/covariance, while upstream updates track lifetime.
            frozen = (self.mean.copy(), self.covariance.copy()) if new_track.observation.partial else None
            self.accept_observation(new_track)
            super().update(new_track, frame_id)
            if frozen is not None:
                self.mean, self.covariance = frozen

        def re_activate(self, new_track, frame_id, new_id=False):
            frozen = (self.mean.copy(), self.covariance.copy()) if new_track.observation.partial else None
            self.accept_observation(new_track)
            super().re_activate(new_track, frame_id, new_id)
            if frozen is not None:
                self.mean, self.covariance = frozen

    class PersonBOTSORT(BOTSORT):
        def init_track(self, results, img=None):
            return [PersonTrack(np.r_[box, i], obs, i)
                    for box, obs, i in zip(results.xywh, results.items, results.indices)]

        def get_dists(self, tracks, detections):
            costs = np.ones((len(tracks), len(detections)), np.float32)
            for i, tr in enumerate(tracks):
                box = np.asarray(tr.tlwh).copy()
                box[2:] += box[:2]
                for j, det in enumerate(detections):
                    hint = det.observation.hint_local_id
                    if det.observation.partial and hint >= 0 and hint != tr.track_id:
                        continue
                    other = np.asarray(det.tlwh).copy()
                    other[2:] += other[:2]
                    overlap = iou(box, other)
                    scale = max(20., box[3] - box[1], other[3] - other[1])
                    distance = float(np.linalg.norm((box[:2] + box[2:] - other[:2] - other[2:]) / 2)) / scale
                    gap = max(1, self.frame_id - tr.frame_id)
                    if distance > min(2., .65 + .035 * gap) and overlap < .01:
                        continue
                    sim = (None if det.observation.partial else
                           cosine(getattr(tr, "smooth_feat", None), getattr(det, "curr_feat", None)))
                    q = det.incoming_quality
                    weight = .70 * min(1., q / .65) if sim is not None else 0.
                    motion = .8 * (1 - overlap) + .2 * min(1., distance)
                    app = min(1., 2 * (1 - sim)) if sim is not None else 1.
                    value = (1 - weight) * motion + weight * app
                    if sim is not None and q >= .45 and sim < .20:
                        value = 1.
                    protected = det.observation.occluded and hint == tr.track_id
                    if gap > max(3, int(.3 * fps)) and not protected and (sim is None or sim < .45):
                        value = 1.
                    costs[i, j] = value
            return costs

    args = SimpleNamespace(track_high_thresh=.25, track_low_thresh=.09, new_track_thresh=cfg.new_track_confidence,
        track_buffer=round(cfg.local_buffer_seconds * fps), match_thresh=.73, fuse_score=False,
        gmc_method="none", proximity_thresh=.5, appearance_thresh=.25, with_reid=False, model="auto")
    kwargs = {"frame_rate": fps} if "frame_rate" in inspect.signature(BOTSORT).parameters else {}
    tracker = PersonBOTSORT(args, **kwargs)
    tracker.max_time_lost = round(cfg.local_buffer_seconds * fps)
    tracker.max_frames_lost = round(cfg.local_buffer_seconds * fps)
    tracker.duplicate_detections_removed = 0
    tracker.lifecycle_config = cfg
    # with_reid=False disables ONLY the upstream encoder. OSNet features are still
    # supplied by our init_track and actively used by our get_dists above.
    return tracker


def track_prediction(track, timestamp):
    """Predict from a full-size measured box, never from a previous prediction.

    A full-size overlapping observation constrains image motion only; appearance
    and floor-coordinate reliability still use the last non-occluded observation.
    """
    if getattr(track, "reliable_time", None) is None:
        return None
    box = getattr(track, "full_box", None)
    stamp = getattr(track, "full_box_time", track.reliable_time)
    if box is None:
        box, stamp = track.reliable_box, track.reliable_time
    age = max(0., timestamp - stamp)
    return box + track.box_velocity * min(age, .4)


def prepare_occlusions(tracker, rows, cfg):
    """Conservative, pre-ReID visibility check. No pose or full-body hallucination.

    Only an unambiguous spatial match may supply a partial detection's association
    box. Raw detection coordinates are never overwritten. Overlap by itself can
    freeze appearance but cannot choose who owns an ambiguous fragment.
    """
    if not rows:
        return
    now = rows[0].timestamp
    candidates = {tr.track_id: tr for tr in tracker.tracked_stracks + tracker.lost_stracks
                  if getattr(tr, "reliable_time", None) is not None
                  and 0 <= now - tr.reliable_time <= cfg.max_gap}
    tracks = list(candidates.values())
    boxes = [track_prediction(tr, now) for tr in tracks]
    costs = np.full((len(rows), len(tracks)), np.inf)
    for i, row in enumerate(rows):
        a = row.bbox
        area = max(1., float(np.prod(a[2:] - a[:2])))
        for j, b in enumerate(boxes):
            intersection = float(np.prod(np.maximum(0., np.minimum(a[2:], b[2:]) - np.maximum(a[:2], b[:2]))))
            coverage = intersection / area
            hratio = (a[3] - a[1]) / max(1., b[3] - b[1])
            score = iou(a, b)
            # Top/bottom fragments inside a predicted body remain candidates.
            if coverage >= .75 and .2 <= hratio < cfg.partial_height_ratio:
                score = max(score, .55 * coverage)
            if score >= .25:
                costs[i, j] = 1 - score
    matches = assign(costs, .75)
    matched = {}
    for i, j in matches:
        alternatives = np.r_[np.delete(costs[i], j), np.delete(costs[:, j], i)]
        if alternatives.size and costs[i, j] + .08 > alternatives.min():
            continue
        matched[i] = j
    overlaps = overlap_ratios([r.bbox for r in rows])
    for i, row in enumerate(rows):
        row.occluded = overlaps[i] >= cfg.occlusion_overlap
        j = matched.get(i)
        if j is not None:
            tr, b = tracks[j], boxes[j]
            wh, old_wh = row.bbox[2:] - row.bbox[:2], np.maximum(1., b[2:] - b[:2])
            row.partial = bool(wh[1] / old_wh[1] < cfg.partial_height_ratio
                               or np.prod(wh) / np.prod(old_wh) < .55)
            row.occluded = row.occluded or row.partial
            row.hint_local_id = int(tr.track_id)
            row.reliable_age = max(0., now - tr.reliable_time)
            if row.partial:
                row.association_bbox = b.copy()
        elif row.occluded:
            # Freeze a likely fragment even when several possible owners remain.
            # Do not fabricate an owner or a corrected association box here.
            row.partial = any(np.isfinite(costs[i, k]) and
                              (row.bbox[3] - row.bbox[1]) < cfg.partial_height_ratio * (b[3] - b[1])
                              for k, b in enumerate(boxes))
        # Birth size is not a rule for erasing a known person's usable evidence.
        if j is None and not bbox_large_enough(row, cfg.birth_min_width, cfg.birth_min_height):
            row.partial = row.occluded = True
        row.overlap = float(overlaps[i])
        row.quality *= 1. - .65 * row.overlap
        row.position_quality *= 1. - .8 * row.overlap
        row.appearance_update_ok = not row.occluded
        # Moderate overlap may still provide a useful *comparison*. Do not write
        # its crop back into a trusted long-term appearance profile.
        if row.partial or row.overlap >= cfg.severe_overlap:
            row.feature = row.color = None
            row.quality = 0.
        # Foot validity is independent of appearance validity. A full-size box
        # with a stable bottom can remain useful on the floor during overlap.
        bottom_jump = (j is not None and abs(row.bbox[3] - boxes[j][3]) >
                       .30 * max(1., boxes[j][3] - boxes[j][1]))
        row.position_usable = bool(not row.partial and not bottom_jump
                                   and row.overlap < cfg.severe_overlap
                                   and row.position_quality >= cfg.position_quality)
        if not row.position_usable:
            row.position_quality = 0.
            row.world = None


def bbox_large_enough(obs, width, height):
    if obs.bbox is None:
        return False
    scale = np.array(obs.image_size, float) / np.array([1280., 720.])
    return bool(np.all((obs.bbox[2:] - obs.bbox[:2]) >= np.array([width, height]) * scale))


def birth_observation_ok(obs, cfg):
    """Stricter evidence for NEW people; fragments may still maintain old tracks."""
    return (not obs.occluded and not obs.partial
            and obs.confidence >= cfg.new_track_confidence
            and obs.quality >= cfg.feature_quality
            and bbox_large_enough(obs, cfg.birth_min_width, cfg.birth_min_height))


def local_candidate_ok(obs, cfg):
    """A recoverable local candidate is not permission to issue a Global ID."""
    return (not obs.partial and obs.overlap < cfg.severe_overlap
            and obs.confidence >= cfg.new_track_confidence
            and obs.quality >= cfg.feature_quality
            and bbox_large_enough(obs, cfg.birth_min_width, cfg.birth_min_height))


def association_observation_ok(obs, cfg):
    # Smaller observations may only maintain an unambiguously identified local
    # owner. They cannot start a track or claim an unrelated global identity.
    return (bbox_large_enough(obs, cfg.min_bbox_width, cfg.min_bbox_height)
            or (obs.hint_local_id >= 0 and obs.reliable_age <= cfg.max_gap
                and bbox_large_enough(obs, 8, 24)))


def position_reliable(obs, cfg):
    usable = not obs.occluded if obs.position_usable is None else obs.position_usable
    return bool(usable and not obs.partial and obs.world is not None
                and obs.position_quality >= cfg.position_quality)


def local_update(tracker, detections, frame):
    cfg = tracker.lifecycle_config
    detections = [d for d in detections if association_observation_ok(d, cfg)]
    for d in detections:
        # Both supported upstream versions check new_track_thresh before activate.
        # Cap only the association score of birth-ineligible boxes: retain them
        # for association, but never let unmatched fragments start a new track.
        d.association_confidence = (d.confidence if local_candidate_ok(d, cfg)
                                    else min(d.confidence, cfg.new_track_confidence - .01))
    tracker.update(FeatureBoxes(detections), frame)
    output, sources = [], set()
    for tr in tracker.tracked_stracks:
        if tr.frame_id != tracker.frame_id:
            continue
        idx = getattr(tr, "source_idx", -1)
        if not 0 <= idx < len(detections) or idx in sources:
            raise RuntimeError("BoT-SORT detection index mismatch; installed API 확인 필요")
        sources.add(idx)
        d = detections[idx]
        d.local_id = int(tr.track_id)
        d.local_confirmed = bool(tr.is_activated)
        prediction = track_prediction(tr, d.timestamp)
        if tr.reliable_time is not None:
            d.reliable_age = max(0., d.timestamp - tr.reliable_time)
        # A confirmed local association with a plausible box can keep its own
        # binding during overlap. It cannot claim a different person's Global ID.
        plausible = prediction is not None and iou(prediction, d.bbox) >= .25
        d.continuity_ok = bool(d.occluded and tr.is_activated and
                              (d.hint_local_id == d.local_id or (d.hint_local_id < 0 and plausible)))
        # Snapshot at this source frame; never read a future track state in the lag buffer.
        if tr.is_activated and (not d.partial or d.continuity_ok):
            feature = getattr(tr, "smooth_feat", None)
            d.track_feature = feature.copy() if feature is not None else None
            d.track_feature_time = tr.feature_time
        d.tracking_bbox = np.asarray(tr.tlwh).copy()
        d.tracking_bbox[2:] += d.tracking_bbox[:2]
        if d.partial:
            d.tracking_bbox = track_prediction(tr, d.timestamp)
        output.append(d)
    if len({d.local_id for d in output}) != len(output):
        raise RuntimeError("동일 카메라/프레임 Local ID 중복")
    return output


def overlap_ratios(boxes):
    """Intersection / EACH box's area; this is not IoU or a depth ordering."""
    result = np.zeros(len(boxes), np.float32)
    for i, a in enumerate(boxes):
        for j in range(i + 1, len(boxes)):
            b = boxes[j]
            area = float(np.prod(np.maximum(0., np.minimum(a[2:], b[2:]) - np.maximum(a[:2], b[:2]))))
            ai, aj = float(np.prod(a[2:] - a[:2])), float(np.prod(b[2:] - b[:2]))
            result[i] = max(result[i], area / max(ai, 1.))
            result[j] = max(result[j], area / max(aj, 1.))
    return result


def clothing_descriptor(crop):
    import cv2
    parts = []
    for lo, hi in ((.15, .55), (.55, .95)):
        part = crop[int(lo * len(crop)):max(int(lo * len(crop)) + 1, int(hi * len(crop)))]
        hsv = cv2.cvtColor(part, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [12, 8], [0, 180, 0, 256]).reshape(-1)
        parts.append(np.sqrt(hist / max(1., float(hist.sum()))))
    return unit(np.concatenate(parts))


def needs_embedding(obs, overlap, tracker, tick, cfg):
    if tick % cfg.reid_interval == 0 or overlap >= .20:
        return True
    known = []
    for tr in tracker.tracked_stracks:
        box = np.asarray(tr.tlwh).copy()
        box[2:] += box[:2]
        if getattr(tr, "smooth_feat", None) is not None:
            known.append((iou(obs.bbox, box), tr))
    known.sort(key=lambda p: p[0], reverse=True)
    if not known or known[0][0] < .55:
        return True
    if len(known) > 1 and known[0][0] - known[1][0] < .15:
        return True
    # Missing or new tracks must obtain fresh appearance before reacquisition.
    return tracker.frame_id - known[0][1].frame_id > 0


def deduplicate_detections(boxes, scores, tracker, timestamp, cfg):
    """Remove near-identical detections, not merely overlapping people.

    Require IoU >= cfg.duplicate_iou, similar dimensions and centers.
    Two plausible confirmed tracks veto suppression. This deliberately leaves
    ambiguous whole-body/fragment pairs for tracking rather than deleting a
    potentially occluded person. Return original indices in detector order.
    """
    boxes, scores = np.asarray(boxes), np.asarray(scores)
    valid = [i for i, (box, score) in enumerate(zip(boxes, scores))
             if np.isfinite(box).all() and np.isfinite(score) and np.all(box[2:] > box[:2])]
    references = {}
    for tr in tracker.tracked_stracks + tracker.lost_stracks:
        if not tr.is_activated or timestamp - tr.last_observation_time > cfg.max_gap:
            continue
        box = np.asarray(tr.tlwh).copy()
        box[2:] += box[:2]
        if np.isfinite(box).all() and np.all(box[2:] > box[:2]):
            references[int(tr.track_id)] = box
    supports = {i: {lid for lid, box in references.items() if iou(boxes[i], box) >= .6}
                for i in valid}
    kept = []
    for i in sorted(valid, key=lambda k: (-float(scores[k]), k)):
        a = boxes[i]
        awh, ac = a[2:] - a[:2], (a[:2] + a[2:]) / 2
        duplicate = False
        for j in kept:
            b = boxes[j]
            bwh, bc = b[2:] - b[:2], (b[:2] + b[2:]) / 2
            near = (iou(a, b) >= cfg.duplicate_iou
                    and np.all(np.minimum(awh, bwh) / np.maximum(awh, bwh) >= .85)
                    and np.all(np.abs(ac - bc) / np.minimum(awh, bwh) <= .08))
            if not near:
                continue
            # Distinct known people may project to nearly the same box.
            separate_tracks = any(x != y for x in supports[i] for y in supports[j])
            if not separate_tracks:
                duplicate = True
                break
        if not duplicate:
            kept.append(i)
    tracker.duplicate_detections_removed += len(valid) - len(kept)
    return np.asarray(sorted(kept), dtype=int)


def detect_batch(model, encoder, packet, calibration, trackers, ticks, args, cfg, timings):
    cams = [cam for cam in (1, 2) if packet.fresh[cam]]
    with timings.measure("yolo_batch"):
        results = model.predict(source=[packet.frames[cam] for cam in cams], classes=[0], conf=.10,
                                iou=.7, imgsz=args.imgsz, device=args.device,
                                verbose=False, **args.yolo_precision)
        # CPU transfer completes these GPU results before the timing closes.
        arrays = [(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()) for r in results]
    if len(arrays) != len(cams):
        raise RuntimeError("YOLO batch result count mismatch")
    detections, crops, feature_targets = {}, [], []
    with timings.measure("crop_and_color"):
        for cam, (boxes, scores) in zip(cams, arrays):
            frame = packet.frames[cam]
            h, w = frame.shape[:2]
            boxes = np.asarray(boxes, np.float32).copy()
            boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, w)
            boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, h)
            keep = deduplicate_detections(boxes, scores, trackers[cam], packet.times[cam], cfg)
            raw_count = len(boxes)
            boxes, scores = boxes[keep], scores[keep]
            rows = []
            for box, score in zip(boxes, scores):
                box = np.asarray(box, np.float32).copy()
                box[[0, 2]] = np.clip(box[[0, 2]], 0, w)
                box[[1, 3]] = np.clip(box[[1, 3]], 0, h)
                x1, y1, x2, y2 = np.round(box).astype(int)
                if x2 - x1 < 8 * w / 1280 or y2 - y1 < 24 * h / 720:
                    continue
                crop = frame[y1:y2, x1:x2]
                boundary = x1 <= 1 or y1 <= 1 or x2 >= w - 1 or y2 >= h - 1
                quality = float(score) * (.65 if boundary else 1.)
                # A separate, conservative proxy for bottom-center reliability.
                position_quality = float(score)
                if y2 >= h - 2:
                    position_quality *= .15
                if x1 <= 1 or x2 >= w - 1:
                    position_quality *= .5
                foot = [(box[0] + box[2]) / 2, box[3]]
                obs = Observation(cam, packet.indices[cam], packet.times[cam], box, float(score),
                                  quality, position_quality, calibration.project(cam, foot, (w, h)),
                                  color=clothing_descriptor(crop), image_size=(w, h))
                rows.append(obs)
            prepare_occlusions(trackers[cam], rows, cfg)
            rows = [obs for obs in rows if association_observation_ok(obs, cfg)]
            packet.diagnostics[cam] = {
                "raw_detections": raw_count, "after_dedup": len(boxes), "after_size_gate": len(rows),
                "occluded": sum(bool(o.occluded) for o in rows),
                "partial": sum(bool(o.partial) for o in rows), "embedding_crops": 0}
            for obs in rows:
                if obs.quality >= cfg.feature_quality and not obs.partial and needs_embedding(
                        obs, obs.overlap, trackers[cam], ticks[cam], cfg):
                    x1, y1, x2, y2 = np.round(obs.bbox).astype(int)
                    crops.append(frame[y1:y2, x1:x2])
                    feature_targets.append(obs)
                    packet.diagnostics[cam]["embedding_crops"] += 1
            detections[cam] = rows
    with timings.measure("osnet_batch"):
        features = encoder.encode(crops)
    if len(features) != len(feature_targets):
        raise RuntimeError("OSNet batch result count mismatch")
    for obs, feat in zip(feature_targets, features):
        obs.feature = unit(feat)
    return detections, len(crops)


class SegmentGuard:
    """Split a reused Local ID at a clear appearance/position discontinuity."""
    def __init__(self, cfg):
        self.cfg, self.state = cfg, {}

    def update(self, observations, now):
        self.state = {k: v for k, v in self.state.items()
                      if now - v[0] <= self.cfg.identity_memory_seconds}
        for d in observations:
            old = self.state.get((d.cam, d.local_id))
            if d.occluded:
                # Occlusion is missing evidence, not an appearance/position jump.
                # Keep segment identity for as long as global appearance memory;
                # expiring at the motion timeout would silently reset the epoch.
                d.epoch = old[1] if old is not None else 0
                continue
            epoch, reference = 0, None
            if old is not None:
                previous_time, epoch, reference, world, pos_q = old
                dt = d.timestamp - previous_time
                sim = cosine(reference, d.feature)
                changed = sim is not None and d.quality >= .45 and sim < .35
                jumped = (world is not None and d.world is not None and min(pos_q, d.position_quality) >= .5
                          and np.linalg.norm(d.world - world) > .9 + self.cfg.max_speed * max(dt, 0.))
                if changed or jumped:
                    epoch += 1
                    reference = None
            d.epoch = epoch
            if d.feature is not None and d.quality >= self.cfg.feature_quality:
                reference = d.feature.copy() if reference is None else unit(.8 * reference + .2 * d.feature)
            self.state[d.cam, d.local_id] = (d.timestamp, epoch, reference, d.world, d.position_quality)


@dataclass
class Descriptor:
    first: Observation
    samples: list
    feature: np.ndarray | None
    color: np.ndarray | None
    feature_from_history: bool = False

    @property
    def cam(self):
        return self.first.cam

    @property
    def key(self):
        return self.first.key

    def supported(self, cfg):
        return (len(self.samples) >= cfg.min_evidence and
                self.samples[-1].timestamp - self.samples[0].timestamp >= cfg.min_evidence_seconds - 1e-8)


def describe(obs, window, cfg):
    unique = {}
    for packet in window:
        for row in packet.observations.get(obs.cam, []):
            if row.key == obs.key and row.timestamp >= obs.timestamp - 1e-8:
                unique[row.frame] = row
    samples = sorted(unique.values(), key=lambda d: d.timestamp)
    if not samples:
        samples = [obs]
    good = [d for d in samples if not d.partial and d.feature is not None and d.quality >= cfg.feature_quality]
    feature = unit(np.average(np.stack([d.feature for d in good]), axis=0,
                              weights=[d.quality for d in good])) if good else None
    from_history = False
    if (feature is None and obs.track_feature is not None and obs.track_feature_time is not None
            and obs.local_confirmed and (not obs.partial or obs.continuity_ok)
            and 0 <= obs.timestamp - obs.track_feature_time <= cfg.feature_history_seconds):
        feature, from_history = obs.track_feature.copy(), True
    colors = [d.color for d in samples if not d.partial and d.color is not None and d.quality >= cfg.feature_quality]
    return Descriptor(obs, samples, feature, unit(np.mean(colors, axis=0)) if colors else None, from_history)


@dataclass
class Identity:
    gid: int
    last: float
    profiles: dict = field(default_factory=dict)
    positions: dict = field(default_factory=dict)
    velocities: dict = field(default_factory=dict)
    keys: dict = field(default_factory=dict)
    archived_at: float | None = None


class GlobalTracker:
    """Fixed-lag evidence + reversible current assignment, no historical group merge."""
    def __init__(self, cfg, event_sink=None):
        self.cfg, self.event_sink = cfg, event_sink
        self.identities, self.bindings, self.last_emitted = {}, {}, {}
        self.birth_evidence, self.deferred_reasons = {}, {}
        self.next_gid = 1
        self.events = Counter()
        self.defer_counts = Counter()
        # Independent observations: these update even when no Global ID is assigned.
        # Store source-frame snapshots, never references into the lookahead buffer.
        self.motion_history = {}
        self.motion_recoveries = {}
        self.current_motion_observations = []
        self.binding_since = {}
        self.frame_bindings = {}
        self.frame_owners = {}
        self.cross_recoveries = {}

    def conflict_point(self, obs):
        """Comparison evidence is weaker than permission to update an anchor.

        Return (point, uncertainty). Never use a future sample or extrapolate a
        hidden box. Missing evidence is not itself a contradictory observation.
        """
        if (position_reliable(obs, self.cfg) and not obs.occluded
                and obs.position_anchor_reason != 'bbox_size_jump'):
            return obs.world, 0.
        history = self.motion_history.get(obs.key, [])
        for row in reversed(history):
            age = obs.timestamp - row['time']
            if age < -1e-8:
                continue
            if age > self.cfg.conflict_history_seconds:
                break
            if row['clean'] and row['world'] is not None:
                return row['world'], self.cfg.max_speed * max(0., age)
            if not row['continuous']:
                break
        return None

    def position_conflict(self, a, b):
        if abs(a.timestamp - b.timestamp) > self.cfg.sync_tolerance + 1e-8:
            return None
        x, y = self.conflict_point(a), self.conflict_point(b)
        if x is None or y is None:
            return None
        distance = float(np.linalg.norm(x[0] - y[0]))
        radius = (1.5 * self.cfg.cross_distance + x[1] + y[1]
                  + self.cfg.max_speed * abs(a.timestamp - b.timestamp))
        return dict(distance_m=distance, radius_m=radius) if distance > radius else None

    def prepare_owners(self, descriptors, now):
        """Freeze ownership BEFORE either camera assigns or updates a profile."""
        self.frame_bindings = self.bindings.copy()
        self.frame_owners = {}
        self.cross_recoveries = {}
        for cam, rows in descriptors.items():
            for d in rows:
                gid = self.frame_bindings.get(d.key)
                identity = self.identities.get(gid)
                if (identity is None or identity.archived_at is not None
                        or identity.keys.get(cam) != d.key or not d.first.local_confirmed
                        or now - identity.last > self.cfg.cross_owner_seconds):
                    continue
                point = self.conflict_point(d.first)
                ref = identity.positions.get(cam)
                sim = cosine(d.feature, identity.profiles.get(cam))
                if (point is None or ref is None or sim is None
                        or sim < self.cfg.cross_owner_similarity
                        or not 0. <= d.first.timestamp - ref[1] <= self.cfg.cross_owner_seconds):
                    continue
                age = d.first.timestamp - ref[1]
                pred = ref[0] + identity.velocities.get(cam, np.zeros(2)) * min(age, .4)
                if np.linalg.norm(point[0] - pred) > .7 + self.cfg.max_speed * age + point[1]:
                    continue
                history = self.motion_history.get(d.key, [])
                votes = [h for h in history if h['clean'] and h['reference_gid'] == gid
                         and 0. <= d.first.timestamp - h['time'] <= self.cfg.cross_owner_seconds]
                if (len(votes) < self.cfg.min_evidence
                        or votes[-1]['time'] - votes[0]['time'] < self.cfg.min_evidence_seconds - 1e-8):
                    continue
                since = self.binding_since.get(d.key, (gid, votes[0]['time']))
                self.frame_owners[cam, gid] = dict(descriptor=d, since=since[1], similarity=sim)
        # When an earlier wrong link already exists, protect the longer established
        # owner, not whichever camera happens to execute first. Equal-age conflicts
        # remain subject to the existing pair/ambiguity checks, without a forced winner.
        for gid in self.identities:
            a, b = self.frame_owners.get((1, gid)), self.frame_owners.get((2, gid))
            if a and b and self.position_conflict(a['descriptor'].first, b['descriptor'].first):
                if abs(a['since'] - b['since']) >= self.cfg.min_evidence_seconds:
                    loser = a if a['since'] > b['since'] else b
                    del self.frame_owners[loser['descriptor'].cam, gid]

    def owner_conflict(self, d, gid):
        owner = self.frame_owners.get((3 - d.cam, gid))
        if owner is None:
            return None
        own = self.frame_owners.get((d.cam, gid))
        if own is not None and own['descriptor'].key == d.key:
            return None
        conflict = self.position_conflict(d.first, owner['descriptor'].first)
        if conflict is not None:
            return dict(conflict, owner_camera=3-d.cam,
                        owner_local_id=owner['descriptor'].first.local_id)
        return None

    def prepare_cross_recoveries(self, groups):
        """Only an unambiguous measured pair can repair a camera-specific anchor."""
        cfg = self.cfg
        for group in groups:
            if len(group) != 2:
                continue
            for d, other in (group, group[::-1]):
                gid = self.frame_bindings.get(other.key)
                owner = self.frame_owners.get((other.cam, gid))
                if (owner is None or owner['descriptor'].key != other.key
                        or not self.anchor_observation_ok(d.first)
                        or not self.anchor_observation_ok(other.first)
                        or d.feature_from_history or other.feature_from_history):
                    continue
                # A healthy owner does not need a reset; preserve its velocity EMA.
                own = self.frame_owners.get((d.cam, gid))
                if own is not None and own['descriptor'].key == d.key:
                    continue
                old_gid = self.frame_bindings.get(d.key)
                old_owner = self.frame_owners.get((d.cam, old_gid))
                if old_gid != gid and old_owner is not None and old_owner['descriptor'].key == d.key:
                    continue
                sim = cosine(d.feature, other.feature)
                if sim is None or sim < max(.62, cfg.cross_similarity):
                    continue
                if np.linalg.norm(d.first.world - other.first.world) > cfg.cross_distance:
                    continue
                used, votes = set(), []
                for a in d.samples:
                    if (not position_reliable(a, cfg) or a.occluded
                            or a.position_quality < cfg.position_anchor_quality):
                        continue
                    options = [b for b in other.samples if b.frame not in used
                               and position_reliable(b, cfg) and not b.occluded
                               and b.position_quality >= cfg.position_anchor_quality
                               and abs(a.timestamp-b.timestamp) <= cfg.sync_tolerance + 1e-8]
                    if not options:
                        continue
                    b = min(options, key=lambda row: abs(a.timestamp-row.timestamp))
                    used.add(b.frame)
                    votes.append((a.timestamp, a.world, b.world))
                if (len(votes) < cfg.min_evidence
                        or votes[-1][0]-votes[0][0] < cfg.min_evidence_seconds-1e-8):
                    continue
                offsets = np.array([a-b for _, a, b in votes])
                distances = np.linalg.norm(offsets, axis=1)
                if (np.mean(distances <= cfg.cross_distance) < .8
                        or np.median(distances) > cfg.cross_distance
                        or np.max(np.linalg.norm(offsets-np.median(offsets, axis=0), axis=1)) > .5*cfg.cross_distance):
                    continue
                da, db = votes[-1][1]-votes[0][1], votes[-1][2]-votes[0][2]
                if min(np.linalg.norm(da), np.linalg.norm(db)) > .2 and np.dot(da, db) <= 0:
                    continue
                self.cross_recoveries[d.key, gid] = dict(
                    owner_camera=other.cam, owner_local_id=other.first.local_id,
                    samples=len(votes), distance_m=float(np.linalg.norm(d.first.world-other.first.world)))

    def anchor_observation_ok(self, obs):
        return bool(position_reliable(obs, self.cfg) and not obs.occluded
                    and obs.position_quality >= self.cfg.position_anchor_quality
                    and obs.position_anchor_ok is not False)

    def motion_comparison_ok(self, obs):
        # A true image jump is contradictory evidence, not missing geometry.
        return bool(self.anchor_observation_ok(obs) or (
            obs.position_anchor_reason == 'image_discontinuity' and
            position_reliable(obs, self.cfg) and not obs.occluded and
            obs.position_quality >= self.cfg.position_anchor_quality))

    def record_motion(self, obs):
        cfg = self.cfg
        history = self.motion_history.setdefault(obs.key, deque())
        if history and obs.frame == history[-1]['frame']:
            obs.position_anchor_ok = history[-1]['clean']
            obs.position_anchor_reason = history[-1]['reason']
            return
        if history and (obs.timestamp <= history[-1]['time'] or
                        obs.timestamp-history[-1]['time'] > cfg.motion_sample_gap):
            history.clear()
        previous = history[-1] if history else None
        clean = bool(position_reliable(obs, cfg) and not obs.occluded and obs.bbox is not None
                     and obs.position_quality >= cfg.position_anchor_quality)
        reason = 'clean' if clean else 'occluded_partial_or_low_quality'
        continuous = False
        if previous is not None and previous['box'] is not None and obs.bbox is not None:
            a, b = previous['box'], obs.bbox
            old_wh, wh = np.maximum(1., a[2:]-a[:2]), np.maximum(1., b[2:]-b[:2])
            scale = max(20., old_wh[1], wh[1])
            displacement = np.linalg.norm((b[:2]+b[2:]-a[:2]-a[2:])/2)
            continuous = bool(iou(a, b) >= .10 and displacement <= .45*scale)
            ratio = wh/old_wh
            stable_shape = bool(.80 <= ratio[1] <= 1.25 and .65 <= ratio[0] <= 1.55)
            # An abruptly returning full body is not yet a trusted footpoint.
            # The next stable detections can establish a new anchor independently.
            if clean and (not continuous or not stable_shape):
                clean = False
                reason = 'image_discontinuity' if not continuous else 'bbox_size_jump'
        obs.position_anchor_ok, obs.position_anchor_reason = clean, reason
        history.append(dict(frame=obs.frame, time=obs.timestamp,
            box=None if obs.bbox is None else obs.bbox.copy(),
            world=None if obs.world is None else obs.world.copy(),
            feature=None if obs.feature is None else obs.feature.copy(),
            quality=obs.quality, clean=clean, continuous=continuous, reason=reason,
            reference_gid=previous.get('reference_gid') if previous and continuous else None))
        while history and obs.timestamp-history[0]['time'] > cfg.motion_history_seconds:
            history.popleft()

    def recover_motion(self, d, identity):
        """Validate a new anchor for this SAME binding. Does not commit anything.

        No lookahead/history-EMA appearance votes, no inference from elapsed wait.
        Occupancy and other-camera contradictions are still checked by resolve.
        """
        cfg, obs = self.cfg, d.first
        def deny(reason):
            obs.match_details['motion_recovery'] = {'gid': identity.gid, 'status': 'rejected', 'reason': reason}
            return None
        if (self.bindings.get(d.key) != identity.gid or identity.archived_at is not None
                or not obs.local_confirmed or not self.anchor_observation_ok(obs)
                or identity.keys.get(d.cam) != d.key):
            return deny('not_current_owner_or_untrusted_foot')
        profile = identity.profiles.get(d.cam)
        if profile is None:
            return deny('camera_profile_missing')
        history = self.motion_history.get(d.key, [])
        if (not history or history[-1]['frame'] != obs.frame
                or history[-1]['reference_gid'] != identity.gid):
            return deny('image_chain_not_connected')
        run = []
        for sample in reversed(history):
            if not sample['clean']:
                break
            run.append(sample)
            if not sample['continuous']:
                break
        run.reverse()
        # A short RECENT stable run is enough; old noise must not veto recovery forever.
        selected = None
        for count in range(cfg.motion_recovery_samples, len(run)+1):
            recent = run[-count:]
            fresh = [r for r in recent if r['feature'] is not None and r['quality'] >= cfg.position_anchor_quality]
            if (len(fresh) >= 2 and recent[-1]['time']-recent[0]['time'] >= cfg.motion_recovery_seconds-1e-8):
                selected = recent, fresh
                break
        if selected is None:
            return deny('insufficient_fresh_stable_observations')
        run, fresh = selected
        sims = [cosine(r['feature'], profile) for r in fresh]
        if len(sims) < 2 or any(s is None or s < cfg.motion_recovery_similarity for s in sims):
            return deny('appearance_not_strong_enough')
        stamps = np.array([r['time'] for r in run])
        points = np.array([r['world'] for r in run])
        velocities = np.diff(points, axis=0)/np.diff(stamps)[:, None]
        velocity = np.median(velocities, axis=0)
        if np.max(np.linalg.norm(velocities, axis=1)) > cfg.motion_recovery_speed:
            return deny('local_speed_outlier')
        residuals = np.diff(points, axis=0) - np.diff(stamps)[:, None]*velocity
        if np.max(np.linalg.norm(residuals, axis=1)) > .20:
            return deny('local_motion_inconsistent')
        # Check all current owners BEFORE assignment order can hide a conflict.
        for other_obs in self.current_motion_observations:
            if (other_obs.key == obs.key or self.bindings.get(other_obs.key) != identity.gid
                    or not self.motion_comparison_ok(other_obs)
                    or abs(other_obs.timestamp-obs.timestamp) > cfg.sync_tolerance):
                continue
            if other_obs.cam != obs.cam:
                if np.linalg.norm(other_obs.world-obs.world) > 1.5*cfg.cross_distance:
                    return deny('cross_camera_position_conflict')
            else:
                state = identity.positions.get(obs.cam)
                similarity = cosine(other_obs.feature, profile)
                if state is not None and similarity is not None and similarity >= .55:
                    dt = max(0., other_obs.timestamp-state[1])
                    prediction = state[0]+identity.velocities.get(obs.cam, np.zeros(2))*min(dt, .4)
                    if np.linalg.norm(other_obs.world-prediction) <= .7+cfg.max_speed*min(dt, cfg.max_gap):
                        return deny('same_camera_owner_supported')
        # Do not re-anchor a locally switched person when another live identity
        # explains the observations with comparable appearance and valid geometry.
        own_sim = float(np.mean(sims))
        for other in self.identities.values():
            if other.gid == identity.gid or other.archived_at is not None:
                continue
            refs = ([other.profiles[d.cam]] if d.cam in other.profiles else list(other.profiles.values()))
            rivals = [cosine(d.feature, p) for p in refs]
            rivals = [s for s in rivals if s is not None]
            if (rivals and max(rivals) >= own_sim-cfg.ambiguity_margin
                    and self.identity_cost([d], other, obs.timestamp) <= cfg.identity_cost_limit):
                return deny('competing_live_identity')
        obs.match_details['motion_recovery'] = {'gid': identity.gid, 'status': 'candidate_ready',
            'samples': len(run), 'fresh_features': len(sims), 'similarity': own_sim}
        return dict(world=obs.world.copy(), velocity=velocity, samples=len(run),
                    fresh_features=len(sims), similarity=own_sim, span_s=float(stamps[-1]-stamps[0]))

    def event(self, kind, timestamp, **details):
        self.events[kind] += 1
        if self.event_sink:
            self.event_sink({"type": kind, "time_s": timestamp, **details})

    def protected(self, d, identity):
        obs = d.first
        return (self.bindings.get(d.key) == identity.gid and obs.occluded
                and obs.continuity_ok and obs.reliable_age <= self.cfg.max_gap)

    def cross_conflict(self, a, b):
        """Contradictory measured evidence differs from simply missing evidence."""
        sim = cosine(a.feature, b.feature)
        if sim is not None and sim < .20:
            return True
        return self.position_conflict(a.first, b.first) is not None

    def record_evidence(self, d):
        """Past distinct observations only: future lookahead cannot publish early."""
        obs, cfg = d.first, self.cfg
        history = self.birth_evidence.setdefault(d.key, deque())
        while history and obs.timestamp - history[0][1] > cfg.birth_window_seconds:
            history.popleft()
        if not history or history[-1][0] != obs.frame:
            history.append((obs.frame, obs.timestamp, birth_observation_ok(obs, cfg)))

    def birth_ready(self, d):
        if not d.first.local_confirmed or not birth_observation_ok(d.first, self.cfg):
            return False
        good = [stamp for _, stamp, valid in self.birth_evidence.get(d.key, []) if valid]
        return (len(good) >= self.cfg.birth_min_hits and
                good[-1] - good[0] >= self.cfg.birth_min_seconds - 1e-8 and d.feature is not None)

    def defer(self, d, reason, now):
        d.first.decision = reason
        self.defer_counts[reason] += 1
        self.events["deferred_observations"] += 1
        if self.deferred_reasons.get(d.key) != reason:
            self.event("identity_deferred", now, camera=d.cam, local_id=d.first.local_id,
                       segment_epoch=d.first.epoch, reason=reason, previous_gid=self.bindings.get(d.key))
            self.deferred_reasons[d.key] = reason

    def pair_cost(self, a, b):
        cfg = self.cfg
        same = self.bindings.get(a.key) is not None and self.bindings.get(a.key) == self.bindings.get(b.key)
        if (a.first.occluded or b.first.occluded) and same:
            identity = self.identities.get(self.bindings.get(a.key)) if same else None
            # Missing evidence may preserve an established link. Otherwise use
            # measured evidence from the fixed-lag window below, even on overlap.
            if (identity is not None and not self.cross_conflict(a, b)
                    and all(not d.first.occluded or self.protected(d, identity) for d in (a, b))):
                cost = self.identity_cost([a, b], identity, max(a.first.timestamp, b.first.timestamp))
                if np.isfinite(cost) and cost <= cfg.identity_cost_limit:
                    return .05
        sim = cosine(a.feature, b.feature)
        threshold = .35 if same else cfg.cross_similarity
        if sim is None or sim < threshold:
            return math.inf
        # Nearest source-time pairs, one vote per distinct source frame.
        used, distances, stamp_pairs = set(), [], []
        for x in a.samples:
            options = [y for y in b.samples if y.frame not in used and
                       abs(y.timestamp - x.timestamp) <= cfg.sync_tolerance + 1e-8]
            if not options:
                continue
            y = min(options, key=lambda row: abs(row.timestamp - x.timestamp))
            used.add(y.frame)
            if not position_reliable(x, cfg) or not position_reliable(y, cfg):
                continue
            distances.append(float(np.linalg.norm(x.world - y.world)))
            stamp_pairs.append((x.timestamp + y.timestamp) / 2)
        if not distances:
            return math.inf
        median = float(np.median(distances))
        if median > cfg.cross_distance or np.mean(np.asarray(distances) <= cfg.cross_distance) < .65:
            return math.inf
        if not same:
            if len(distances) < cfg.min_evidence or max(stamp_pairs) - min(stamp_pairs) < cfg.min_evidence_seconds - 1e-8:
                return math.inf
        color = cosine(a.color, b.color)
        value = .50 * (1 - sim) + .45 * median / cfg.cross_distance + .05 * (1 - (color if color is not None else .5))
        return max(0., value - (.06 if same else 0.))

    def identity_cost(self, descriptors, identity, now, audit=None):
        cfg = self.cfg
        def reject(reason):
            if audit is not None:
                audit['reason'] = reason
            return math.inf
        if audit is not None:
            audit.update(reason='ok', gid=identity.gid, archived=identity.archived_at is not None,
                         evidence=[])
        if now - identity.last > cfg.identity_memory_seconds:
            return reject('identity_expired')
        dormant = now - identity.last > cfg.max_gap
        values = []
        for d in descriptors:
            conflict = self.owner_conflict(d, identity.gid)
            if conflict is not None:
                if audit is not None:
                    audit['owner_conflict'] = conflict
                return reject('current_owner_position_conflict')
            if self.protected(d, identity) and (d.feature is None or d.first.partial):
                values.append(-.12)
                continue
            if d.first.partial and not self.protected(d, identity):
                return reject('partial_without_continuity')
            bound = self.bindings.get(d.key) == identity.gid
            profile = identity.profiles.get(d.cam)
            if profile is None and identity.profiles:
                sims = [cosine(d.feature, p) for p in identity.profiles.values()]
                sims = [v for v in sims if v is not None]
                sim = max(sims) if sims else None
                smin = .62
            else:
                sim = cosine(d.feature, profile)
                smin = .35 if bound else .55
            if dormant:
                # A long-lived appearance memory is not long-lived motion proof.
                smin = max(smin, cfg.dormant_similarity)
                if d.feature_from_history or not d.first.local_confirmed:
                    return reject('fresh_appearance_required')
            evidence = {'camera': d.cam, 'local_id': d.first.local_id,
                        'similarity': sim, 'threshold': smin, 'bound': bound}
            if audit is not None:
                audit['evidence'].append(evidence)
            if sim is not None and sim < smin:
                return reject('appearance_below_threshold')
            if sim is None and (not bound or dormant):
                return reject('appearance_missing')
            if not bound:
                # A new Local ID can recover early, but needs actual floor
                # evidence, not merely resemblance to a cached appearance.
                if not d.first.local_confirmed or not any(position_reliable(s, cfg) for s in d.samples):
                    return reject('floor_or_confirmation_missing')
            geo = .5
            position = identity.positions.get(d.cam)
            if position is None and identity.positions:
                other = max(identity.positions, key=lambda cam: identity.positions[cam][1])
                position = identity.positions[other]
                velocity = identity.velocities.get(other)
            else:
                velocity = identity.velocities.get(d.cam)
            # An established binding uses its CURRENT trustworthy measurement.
            # Occluded/unstable feet cannot veto it or a clean future footpoint
            # retroactively reject the present frame. New-ID geometry stays v3.
            measured = ((d.first if self.motion_comparison_ok(d.first) else None) if bound else
                        next((s for s in d.samples if position_reliable(s, cfg)), None))
            cross_recovery = self.cross_recoveries.get((d.key, identity.gid))
            if cross_recovery is not None:
                evidence['cross_view_recovery'] = cross_recovery
                geo = min(1., cross_recovery['distance_m'] / cfg.cross_distance)
                if audit is not None:
                    audit['reason'] = 'cross_view_recovery_supported'
            elif position is not None and measured is not None:
                p, stamp, quality = position
                dt = max(0., measured.timestamp - stamp)
                predicted = p + velocity * min(dt, .4) if velocity is not None and dt <= cfg.max_gap else p
                distance = float(np.linalg.norm(measured.world - predicted))
                radius = .7 + cfg.max_speed * min(dt, cfg.max_gap)
                evidence.update(distance_m=distance, radius_m=radius, position_age_s=dt)
                if dt <= cfg.max_gap and min(quality, measured.position_quality) >= cfg.position_quality and distance > radius:
                    recovery = self.recover_motion(d, identity) if bound else None
                    if recovery is None:
                        return reject('motion_conflict')
                    self.motion_recoveries[d.key, identity.gid] = recovery
                    evidence.update(motion_recovery=True, recovery_samples=recovery['samples'],
                                    recovery_similarity=recovery['similarity'])
                    if audit is not None:
                        audit['reason'] = 'motion_recovery_supported'
                    distance = 0.  # Local measured trajectory validates a new anchor.
                # Old coordinates cannot veto a person who moved while hidden.
                geo = min(1., distance / max(.8, radius)) if dt <= cfg.max_gap else .5
            value = .65 * (1 - (sim if sim is not None else .55)) + .35 * geo
            if bound:
                value -= .12
            # Preserve a negative continuity bonus at zero appearance/position cost.
            # Clamping to zero would create ties and swap perfectly stable IDs.
            values.append(value)
        return float(np.mean(values)) if values else math.inf

    def archive_recovery_ok(self, group, identity):
        """An explicitly superseded number is a fallback, never a duplicate rival."""
        for d in group:
            clean = [s for s in d.samples if not s.occluded and s.appearance_update_ok
                     and s.feature is not None and s.quality >= self.cfg.feature_quality
                     and position_reliable(s, self.cfg)]
            profiles = ([identity.profiles[d.cam]] if d.cam in identity.profiles
                        else list(identity.profiles.values()))
            scores = [cosine(d.feature, p) for p in profiles]
            scores = [s for s in scores if s is not None]
            if not clean or d.feature_from_history or not scores or max(scores) < .85:
                return False
        return True

    def prefer_valid_continuation(self, matrix, units, ids):
        """Reduce only a feasible old assignment's cost; never veto a new one."""
        scores = matrix.copy()
        for i, group in enumerate(units):
            gids = {self.bindings.get(d.key) for d in group}
            if len(gids) != 1:
                continue
            gid = next(iter(gids))
            if gid not in ids:
                continue
            j = ids.index(gid)
            if scores[i, j] <= self.cfg.identity_cost_limit:
                scores[i, j] -= self.cfg.continuity_margin
        return scores

    def update_identity(self, identity, d, now):
        cfg, obs = self.cfg, d.first
        if identity.archived_at is not None:
            identity.archived_at = None
            self.event('identity_restored', now, gid=identity.gid)
        old = identity.positions.get(d.cam)
        recovery = self.motion_recoveries.get((d.key, identity.gid))
        cross_recovery = self.cross_recoveries.get((d.key, identity.gid))
        if self.anchor_observation_ok(obs):
            if cross_recovery is not None:
                # Reset only if this is a changed owner or a contradicting anchor.
                age = max(0., obs.timestamp-old[1]) if old is not None else 0.
                pred = (old[0] + identity.velocities.get(d.cam, np.zeros(2))*min(age, .4)
                        if old is not None else obs.world)
                if (self.bindings.get(d.key) != identity.gid or old is None
                        or np.linalg.norm(obs.world-pred) > .7+cfg.max_speed*min(age, cfg.max_gap)):
                    identity.velocities[d.cam] = np.zeros(2)
                    obs.match_details['cross_view_anchor_reset'] = cross_recovery
                    self.event('cross_view_anchor_reset', now, gid=identity.gid,
                               camera=d.cam, local_id=obs.local_id, source_frame=obs.frame,
                               **cross_recovery)
            elif recovery is not None:
                identity.velocities[d.cam] = recovery['velocity'].copy()
                self.event('motion_anchor_reset', now, gid=identity.gid, camera=d.cam,
                           local_id=obs.local_id, source_frame=obs.frame,
                           previous_world=None if old is None else old[0].tolist(),
                           world=obs.world.tolist(), samples=recovery['samples'],
                           fresh_features=recovery['fresh_features'], similarity=recovery['similarity'])
                obs.match_details['motion_anchor_reset'] = True
            elif old is not None and obs.timestamp > old[1] + 1e-8:
                dt = obs.timestamp - old[1]
                velocity = (obs.world - old[0]) / dt
                if np.linalg.norm(velocity) <= cfg.max_speed:
                    previous = identity.velocities.get(d.cam, velocity)
                    identity.velocities[d.cam] = .8 * previous + .2 * velocity
            identity.positions[d.cam] = (obs.world.copy(), obs.timestamp, obs.position_quality)
        if (not obs.occluded and obs.appearance_update_ok and not d.feature_from_history
                and d.feature is not None):
            clean = [s for s in d.samples if not s.occluded and s.appearance_update_ok
                     and s.feature is not None and s.quality >= cfg.feature_quality]
            if clean:
                trusted = unit(np.average(np.stack([s.feature for s in clean]), axis=0,
                                          weights=[s.quality for s in clean]))
                old_profile = identity.profiles.get(d.cam)
                identity.profiles[d.cam] = trusted if old_profile is None else unit(.90 * old_profile + .10 * trusted)
        identity.keys[d.cam] = d.key
        identity.last = now
        previous_gid = self.bindings.get(d.key)
        if previous_gid is not None and previous_gid != identity.gid:
            self.event("reassignment", now, camera=d.cam, local_id=obs.local_id,
                       from_gid=previous_gid, to_gid=identity.gid)
        if previous_gid != identity.gid or d.key not in self.binding_since:
            self.binding_since[d.key] = (identity.gid, now)
        self.bindings[d.key] = identity.gid
        history = self.motion_history.get(d.key)
        if history and history[-1]['frame'] == obs.frame:
            history[-1]['reference_gid'] = identity.gid
        self.deferred_reasons.pop(d.key, None)

    def resolve(self, packet, window, truncated=False):
        cfg, now = self.cfg, packet.timestamp
        self.motion_recoveries.clear()
        self.current_motion_observations = [obs for rows in packet.observations.values() for obs in rows]
        self.motion_history = {key: history for key, history in self.motion_history.items()
                               if history and now-history[-1]['time'] <= cfg.motion_history_seconds}
        for gid, identity in self.identities.items():
            if now - identity.last > cfg.identity_memory_seconds:
                self.event('identity_expired', now, gid=gid, last_match_s=identity.last)
        self.identities = {gid: v for gid, v in self.identities.items()
                           if now - v.last <= cfg.identity_memory_seconds}
        self.bindings = {key: gid for key, gid in self.bindings.items() if gid in self.identities}
        self.birth_evidence = {key: history for key, history in self.birth_evidence.items()
                               if history and now - history[-1][1] <= cfg.max_gap}
        self.deferred_reasons = {key: reason for key, reason in self.deferred_reasons.items()
                                 if key in self.birth_evidence or key in self.bindings}
        output, occupied, current = {1: [], 2: []}, {1: set(), 2: set()}, {}
        desc = {1: [], 2: []}
        for cam in (1, 2):
            previous = self.last_emitted.get(cam)
            if not packet.fresh[cam] and previous is not None and previous[0] == packet.indices[cam]:
                output[cam] = previous[1]
                for obs, gid, status in output[cam]:
                    if obs.bbox is not None:
                        occupied[cam].add(gid)
                        current[cam, gid] = describe(obs, window, cfg)
                continue
            for obs in packet.observations.get(cam, []):
                obs.match_details = {'previous_gid': self.bindings.get(obs.key), 'attempts': []}
                self.record_motion(obs)
                d = describe(obs, window, cfg)
                self.record_evidence(d)
                identity = self.identities.get(self.bindings.get(d.key))
                if not obs.local_confirmed:
                    self.defer(d, "local_unconfirmed", now)
                    continue
                if identity is not None and obs.partial and not self.protected(d, identity):
                    # Leave the old binding intact without drawing or refreshing
                    # it. Missing evidence must not create a replacement identity.
                    self.defer(d, "occlusion_insufficient_continuity", now)
                    continue
                desc[cam].append(d)

        self.prepare_owners(desc, now)
        costs = np.full((len(desc[1]), len(desc[2])), np.inf)
        for i, a in enumerate(desc[1]):
            for j, b in enumerate(desc[2]):
                costs[i, j] = self.pair_cost(a, b)
        # Ambiguous new pairs are held apart, rather than choosing an arbitrary tie.
        filtered = costs.copy()
        for i in range(len(desc[1])):
            for j in range(len(desc[2])):
                if not np.isfinite(costs[i, j]):
                    continue
                a, b = desc[1][i], desc[2][j]
                established = self.bindings.get(a.key) is not None and self.bindings.get(a.key) == self.bindings.get(b.key)
                alternatives = np.r_[np.delete(costs[i], j), np.delete(costs[:, j], i)]
                if not established and alternatives.size and costs[i, j] + cfg.ambiguity_margin > alternatives.min():
                    filtered[i, j] = math.inf
        pairs = assign(filtered, cfg.pair_cost_limit)
        paired1, paired2 = {i for i, _ in pairs}, {j for _, j in pairs}
        groups = [[desc[1][i], desc[2][j]] for i, j in pairs]
        singles = {1: [[d] for i, d in enumerate(desc[1]) if i not in paired1],
                   2: [[d] for j, d in enumerate(desc[2]) if j not in paired2]}
        fallback = {1: [], 2: []}
        self.prepare_cross_recoveries(groups)

        def allowed(group, identity):
            for d in group:
                if self.owner_conflict(d, identity.gid) is not None:
                    return False
                reserved_key = reservations.get((d.cam, identity.gid))
                if reserved_key is not None and reserved_key != d.key:
                    return False
                if identity.gid in occupied[d.cam]:
                    return False
                other = current.get((3 - d.cam, identity.gid))
                if other is not None:
                    # Do not bypass the ambiguity gate through sequential single-view assignment.
                    established = (self.bindings.get(d.key) == identity.gid and
                                   self.bindings.get(other.key) == identity.gid)
                    if not established or self.cross_conflict(d, other):
                        return False
            return True

        reservations = {}
        reservation_costs = {}
        for cam in (1, 2):
            for d in desc[cam]:
                identity = self.identities.get(self.bindings.get(d.key))
                if identity is not None and now - identity.last <= cfg.max_gap:
                    value = self.identity_cost([d], identity, now)
                    slot = cam, identity.gid
                    if value <= cfg.identity_cost_limit and value < reservation_costs.get(slot, math.inf):
                        reservations[slot] = d.key
                        reservation_costs[slot] = value

        def bind_group(group, identity, new=False):
            linked = len(group) == 2 or any((3 - d.cam, identity.gid) in current for d in group)
            for d in group:
                if not new and d.key not in self.bindings:
                    self.event("identity_recovered", now, camera=d.cam, local_id=d.first.local_id,
                               gid=identity.gid, before_birth_confirmation=not self.birth_ready(d))
                status = "linked" if linked else "single_view"
                if truncated:
                    status += "_tail"
                self.update_identity(identity, d, now)
                d.first.match_details['selected_gid'] = identity.gid
                d.first.decision = status
                output[d.cam].append((d.first, identity.gid, status))
                occupied[d.cam].add(identity.gid)
                current[d.cam, identity.gid] = d
            if linked:
                self.events["cross_link_frames"] += 1
            if new:
                self.event("new_identity", now, gid=identity.gid,
                           cameras=[d.cam for d in group], supported=all(d.supported(cfg) for d in group))

        def match_groups(units, batch_recovery=False):
            # Reserve existing occlusion continuations before appearance-based
            # assignment, so missing evidence cannot win a different identity.
            pending = []
            for group in units:
                gids = {self.bindings.get(d.key) for d in group}
                identity = self.identities.get(next(iter(gids))) if len(gids) == 1 else None
                if (identity is not None and any(self.protected(d, identity) for d in group)
                        and allowed(group, identity)
                        and self.identity_cost(group, identity, now) <= cfg.identity_cost_limit):
                    bind_group(group, identity)
                else:
                    pending.append(group)
            units = pending
            ids = sorted(self.identities)
            matrix = np.full((len(units), len(ids)), np.inf)
            for i, group in enumerate(units):
                raw_costs, audits = {}, {}
                for j, gid in enumerate(ids):
                    identity = self.identities[gid]
                    audit = {}
                    raw_costs[gid] = self.identity_cost(group, identity, now, audit)
                    audits[gid] = audit
                    if allowed(group, identity):
                        matrix[i, j] = raw_costs[gid]
                    else:
                        audit['availability'] = 'reserved_occupied_or_cross_conflict'
                # Superseded IDs are retained for 300 seconds, but cannot block
                # a live identity through duplicate appearance/ambiguity. Even
                # an occupied live match blocks resurrection of its old number.
                live_match = any(raw_costs[gid] <= cfg.identity_cost_limit
                                 and self.identities[gid].archived_at is None for gid in ids)
                for j, gid in enumerate(ids):
                    identity = self.identities[gid]
                    if identity.archived_at is not None and (
                            live_match or not self.archive_recovery_ok(group, identity)):
                        matrix[i, j] = math.inf
                        audits[gid]['availability'] = 'archived_duplicate_or_weak_recovery'
                attempt = {'stage': 'pair' if len(group) == 2 else 'single', 'candidates': [
                    dict(audits[gid], raw_cost=raw_costs[gid] if np.isfinite(raw_costs[gid]) else None,
                         available_cost=float(matrix[i, j]) if np.isfinite(matrix[i, j]) else None)
                    for j, gid in enumerate(ids)]}
                for d in group:
                    d.first.match_details['attempts'].append(attempt)
            # No extra confirmation frames: a clearly better identity or an
            # invalid/blocked old identity is corrected in this same resolve.
            scored = self.prefer_valid_continuation(matrix, units, ids)
            # Do not arbitrarily recover one of several similar remembered people.
            # Gate against original costs, never against already-filtered values.
            filtered_ids = scored.copy()
            for i, group in enumerate(units):
                for j, gid in enumerate(ids):
                    if not np.isfinite(matrix[i, j]):
                        continue
                    established = all(self.bindings.get(d.key) == gid for d in group)
                    if established and now - self.identities[gid].last <= cfg.max_gap:
                        continue
                    alternatives = np.r_[np.delete(scored[i], j), np.delete(scored[:, j], i)]
                    if alternatives.size and scored[i, j] + cfg.ambiguity_margin > alternatives.min():
                        filtered_ids[i, j] = math.inf
            if batch_recovery:
                established = {(i, j) for i, group in enumerate(units) for j, gid in enumerate(ids)
                               if all(self.bindings.get(d.key) == gid for d in group)
                               and now - self.identities[gid].last <= cfg.max_gap}
                matches = recovery_assignment(scored, cfg.identity_cost_limit,
                                              cfg.ambiguity_margin, established)
                for group in units:
                    for d in group:
                        d.first.match_details['batch_recovery'] = True
            else:
                matches = assign(filtered_ids, cfg.identity_cost_limit)
            used = set()
            for i, j in matches:
                bind_group(units[i], self.identities[ids[j]])
                used.add(i)
            for i, group in enumerate(units):
                if i in used:
                    continue
                if len(group) == 2 and any(d.key in self.bindings for d in group):
                    # A rejected cross-camera proposal must not hide an otherwise
                    # valid established observation in the other camera.
                    for d in group:
                        if cfg.batch_pair_fallback:
                            fallback[d.cam].append([d])
                        else:
                            match_groups([[d]])
                    continue
                if any(d.key in self.bindings for d in group):
                    for d in group:
                        self.defer(d, "existing_identity_match_rejected", now)
                    continue
                if np.any(matrix[i] <= cfg.identity_cost_limit):
                    for d in group:
                        self.defer(d, "ambiguous_existing_identity", now)
                    continue
                if not all(self.birth_ready(d) for d in group):
                    for d in group:
                        reason = "awaiting_confirmation" if birth_observation_ok(d.first, cfg) else "unsafe_new_observation"
                        self.defer(d, reason, now)
                    continue
                identity = Identity(self.next_gid, now)
                self.next_gid += 1
                self.identities[identity.gid] = identity
                bind_group(group, identity, new=True)

        match_groups(groups)
        for cam in (1, 2):
            units = singles[cam] + fallback[cam]
            if cfg.batch_pair_fallback:
                units.sort(key=lambda group: group[0].key)
            match_groups(units, batch_recovery=bool(fallback[cam]))
        active_keys = {d.key for values in desc.values() for d in values}
        # Remove obsolete Local-ID bindings once no active identity references them.
        live_keys = {key for identity in self.identities.values() for key in identity.keys.values()}
        self.bindings = {key: gid for key, gid in self.bindings.items() if key in active_keys or key in live_keys}
        self.binding_since = {key: value for key, value in self.binding_since.items()
                              if key in self.bindings}
        # Supersession changes eligibility, not the last-match lifetime. This
        # archive is still recoverable with distinct fresh evidence; only the
        # expiry check at the START of resolve is allowed to delete an identity.
        for gid, identity in list(self.identities.items()):
            if identity.keys and not any(self.bindings.get(key) == gid for key in identity.keys.values()):
                if identity.archived_at is None:
                    identity.archived_at = now
                    self.event("identity_archived", now, gid=gid, last_match_s=identity.last)
        for cam, rows in output.items():
            if len({gid for _, gid, _ in rows}) != len(rows):
                raise RuntimeError("동일 카메라 출력에 중복 Global ID")
            self.last_emitted[cam] = (packet.indices[cam], rows)
        return output


class FixedLagBuffer:
    """No more than ceil(delay*fps)+2 full frame pairs; actual decision uses lookahead."""
    def __init__(self, delay, fps):
        self.delay, self.fps = delay, fps
        self.packets = deque()
        self.max_packets = math.ceil(delay * fps) + 2
        self.peak = 0

    def push(self, packet):
        self.packets.append(packet)
        self.peak = max(self.peak, len(self.packets))
        if len(self.packets) > self.max_packets:
            raise RuntimeError("고정 지연 버퍼 초과: 출력 drain 로직 확인 필요")

    def ready(self):
        return bool(self.packets and self.packets[-1].timestamp - self.packets[0].timestamp >= self.delay - 1e-8)

    def pop(self, tracker, flush=False):
        if not self.packets or not (flush or self.ready()):
            return None
        first = self.packets[0]
        # Discrete sampling may read one sample beyond the horizon; never use it as evidence.
        window = [p for p in self.packets if p.timestamp <= first.timestamp + self.delay + 1e-8]
        truncated = flush and self.packets[-1].timestamp < first.timestamp + self.delay - 1e-8
        labels = tracker.resolve(first, window, truncated=truncated)
        lookahead = window[-1].timestamp - first.timestamp
        self.packets.popleft()
        return first, labels, lookahead, truncated


def gid_color(gid):
    import cv2
    hsv = np.uint8([[[(gid * 137) % 180, 210, 245]]])
    return tuple(int(v) for v in cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0])


def render(frame, labels, cam, index, stamp):
    import cv2
    out = frame.copy()
    for obs, gid, status in labels:
        if obs.bbox is None:
            continue
        color = gid_color(gid)
        box = np.asarray(obs.bbox).copy()
        box[[0, 2]] = np.clip(box[[0, 2]], 0, out.shape[1] - 1)
        box[[1, 3]] = np.clip(box[[1, 3]], 0, out.shape[0] - 1)
        x1, y1, x2, y2 = np.round(box).astype(int)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 3)
        label = f"ID {gid}"
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, .8, 2)
        ly = max(th + 7, y1 - 5)
        cv2.rectangle(out, (max(0, x1), ly - th - 5), (max(0, x1) + tw + 8, ly + baseline), color, -1)
        cv2.putText(out, label, (max(0, x1) + 4, ly), cv2.FONT_HERSHEY_SIMPLEX, .8, (255, 255, 255), 2)
    cv2.putText(out, f"CAM{cam} | src {index} | {stamp:.3f}s", (12, 25),
                cv2.FONT_HERSHEY_SIMPLEX, .65, (255, 255, 255), 2)
    return out


def combine_frames(frames, display_width):
    import cv2
    images = [cv2.resize(frame, (display_width, max(2, round(frame.shape[0] * display_width / frame.shape[1]))))
              for frame in frames]
    height = max(x.shape[0] for x in images)
    height += height % 2
    return np.hstack([cv2.copyMakeBorder(x, 0, height - x.shape[0], 0, 0, cv2.BORDER_CONSTANT) for x in images])


def make_writer(path, fps, size):
    import cv2
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    if not writer.isOpened():
        writer.release()
        raise RuntimeError(f"영상 writer 생성 실패: {path}")
    return writer


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--video1", default=VIDEO1_PATH)
    p.add_argument("--video2", default=VIDEO2_PATH)
    p.add_argument("--model", default="yolo26n.pt")
    p.add_argument("--output-dir", default="MTMCT_delayed_results_v4-2-2")
    p.add_argument("--start-frame", type=int, default=0)
    p.add_argument("--end-frame", type=int, default=None)
    p.add_argument("--reid-weights", default=REID_WEIGHTS_PATH)
    p.add_argument("--reid-sha256", default=None)
    p.add_argument("--device", default="auto")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--precision", choices=("auto", "fp32", "fp16"), default="auto")
    p.add_argument("--reid-interval", type=int, default=2)
    p.add_argument("--delay-seconds", type=float, default=.3)
    p.add_argument("--cross-distance", type=float, default=1.2)
    p.add_argument("--sync-tolerance", type=float, default=.08)
    p.add_argument("--max-gap", type=float, default=3.)
    p.add_argument("--identity-memory-seconds", type=float, default=300.)
    p.add_argument("--continuity-margin", type=float, default=.04,
                   help="유효한 기존 ID의 작은 비용 우대. 0이면 비활성화; 전환 대기 없음")
    p.add_argument("--disable-batch-fallback", action="store_true",
                   help="비교 실험용: 거절된 카메라 쌍을 v3처럼 한 명씩 재시도")
    p.add_argument("--min-bbox-width", type=int, default=24)
    p.add_argument("--min-bbox-height", type=int, default=80)
    p.add_argument("--birth-min-width", type=int, default=32)
    p.add_argument("--birth-min-height", type=int, default=120)
    p.add_argument("--new-track-conf", type=float, default=.50)
    p.add_argument("--birth-min-hits", type=int, default=3)
    p.add_argument("--birth-min-seconds", type=float, default=.10)
    p.add_argument("--duplicate-iou", type=float, default=.85)
    p.add_argument("--temporal-correction", choices=("none", "auto"), default="none")
    p.add_argument("--cam2-offset-seconds", type=float, default=0.)
    p.add_argument("--allow-resize-calibration", action="store_true")
    p.add_argument("--offline", action="store_true")
    p.add_argument("--show-preview", action="store_true")
    p.add_argument("--pace-input", action="store_true")
    p.add_argument("--no-video", action="store_true")
    p.add_argument("--save-separate", action="store_true")
    p.add_argument("--display-width", type=int, default=720)
    p.add_argument("--validate-only", action="store_true")
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--test-botsort", action="store_true")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.self_test:
        return self_test(args.test_botsort)
    import cv2
    if not (.1 <= args.delay_seconds <= 2. and .1 <= args.cross_distance <= 5.
            and .005 <= args.sync_tolerance <= .5 and .2 <= args.max_gap <= 10.
            and 1 <= args.reid_interval <= 10 and 256 <= args.imgsz <= 2048
            and 64 <= args.display_width <= 1920 and args.display_width % 2 == 0
            and np.isfinite(args.cam2_offset_seconds)):
        raise ValueError("옵션 범위 오류: delay .1~2s, cross-distance .1~5m, sync .005~.5s, "
                         "max-gap .2~10s, reid-interval 1~10, imgsz 256~2048, 짝수 display-width 64~1920")
    cfg = Config(delay_seconds=args.delay_seconds, cross_distance=args.cross_distance,
                 sync_tolerance=args.sync_tolerance, max_gap=args.max_gap,
                 reid_interval=args.reid_interval, local_buffer_seconds=args.max_gap,
                 min_bbox_width=args.min_bbox_width, min_bbox_height=args.min_bbox_height,
                 birth_min_width=args.birth_min_width, birth_min_height=args.birth_min_height,
                 new_track_confidence=args.new_track_conf, birth_min_hits=args.birth_min_hits,
                 birth_min_seconds=args.birth_min_seconds, duplicate_iou=args.duplicate_iou)
    if not np.isfinite(args.identity_memory_seconds) or not 300. <= args.identity_memory_seconds <= 3600:
        raise ValueError("identity-memory-seconds는 300초 이상, 3600초 이하여야 합니다.")
    cfg.identity_memory_seconds = args.identity_memory_seconds
    if not np.isfinite(args.continuity_margin) or not 0. <= args.continuity_margin <= .15:
        raise ValueError("continuity-margin은 0~0.15 범위여야 합니다.")
    cfg.continuity_margin = args.continuity_margin
    cfg.batch_pair_fallback = not args.disable_batch_fallback
    if not (4 <= cfg.min_bbox_width <= cfg.birth_min_width <= 640
            and 12 <= cfg.min_bbox_height <= cfg.birth_min_height <= 720
            and .25 <= cfg.new_track_confidence <= .99 and 2 <= cfg.birth_min_hits <= 10
            and .03 <= cfg.birth_min_seconds <= cfg.birth_window_seconds and .5 <= cfg.duplicate_iou <= .99):
        raise ValueError("신규 확정 옵션 범위 오류: 최소 크기<=신규 크기, conf .25~.99, hits 2~10, seconds .03~.5, IoU .5~.99")
    calibration = Calibration()
    if args.validate_only:
        print(json.dumps(calibration.report, ensure_ascii=False, indent=2))
        return 0
    correction = ("none",)
    names = (Path(args.video1).name, Path(args.video2).name)
    if args.temporal_correction == "auto":
        if names not in TEMPORAL_CORRECTIONS:
            raise ValueError(f"시간 보정표에 없는 영상: {names}. none 또는 수동 offset 사용")
        correction = TEMPORAL_CORRECTIONS[names]
    elif names in TEMPORAL_CORRECTIONS and TEMPORAL_CORRECTIONS[names] != ("none",):
        print("[시간] 현재 보정 none. 해당 파일명의 기존 보정표는 --temporal-correction auto로 적용 가능합니다.")
    run_dir = Path(args.output_dir) / datetime.now().strftime("run_%Y%m%d_%H%M%S_%f")
    run_dir.mkdir(parents=True, exist_ok=False)
    write_json(run_dir / "calibration.json", calibration.report)
    timings, writers = Timings(), {}
    reader = csv_stream = event_stream = diagnostic_stream = None
    manager = buffer = None
    trackers = {}
    model_info = {}
    processed = rendered = crop_count = tail_frames = 0
    observed_total = identified_total = 0
    maximum_age = maximum_backlog = maximum_pair_skew = loop_compute = age_sum = 0.
    status, failure = "failed", None
    started = time.perf_counter()
    processing_start = None
    try:
        with timings.measure("video_init"):
            reader = VideoPairReader((args.video1, args.video2), args.start_frame, args.end_frame,
                                     correction, args.cam2_offset_seconds)
            sizes = {cam: (m["width"], m["height"]) for cam, m in reader.info.items()}
            for cam in (1, 2):
                calibration.check_size(cam, sizes[cam], args.allow_resize_calibration)
            if args.save_separate and any(w % 2 or h % 2 for w, h in sizes.values()):
                raise ValueError("분리 영상 저장에는 짝수 가로/세로 해상도가 필요합니다.")
        os.environ["YOLO_AUTOINSTALL"] = "false"
        if args.offline:
            os.environ["YOLO_OFFLINE"] = "true"
            if not Path(args.model).is_file():
                raise FileNotFoundError(f"오프라인 YOLO 가중치 없음: {args.model}")
        with timings.measure("models_init"):
            import torch
            from ultralytics import YOLO
            from ultralytics.cfg import DEFAULT_CFG_DICT
            args.device = ("cuda:0" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
            if args.device.isdigit():
                args.device = "cuda:" + args.device
            if args.device.startswith("cuda") and not torch.cuda.is_available():
                raise RuntimeError("CUDA 사용 불가; GPU 환경 확인 또는 --device cpu")
            args.half = args.precision != "fp32" and args.device.startswith("cuda")
            args.yolo_precision = ({"quantize": 16 if args.half else 32} if "quantize" in DEFAULT_CFG_DICT
                                   else {"half": args.half})
            if args.precision == "fp16" and not args.half:
                raise ValueError("이 버전의 fp16 옵션은 CUDA에서만 지원합니다.")
            if args.device == "cpu":
                torch.set_num_threads(min(4, os.cpu_count() or 1))
            model = YOLO(args.model)
            if model.task != "detect":
                raise ValueError("bbox 전용 detection 모델을 지정하세요. 기본 yolo26n.pt (Pose/Seg 불가)")
            weight_path, weight_hash = prepare_weights(args.reid_weights, args.offline, args.reid_sha256)
            encoder = FeatureEncoder(weight_path, args.device, torch, half=args.half)
            trackers = {cam: make_local_tracker(reader.fps, cfg) for cam in (1, 2)}
            versions = {}
            for pkg in ("torch", "torchvision", "ultralytics", "torchreid", "numpy", "lap"):
                try:
                    versions[pkg] = importlib.metadata.version(pkg)
                except importlib.metadata.PackageNotFoundError:
                    pass
            model_info = {"yolo": args.model, "task": model.task, "device": args.device,
                          "fp16": args.half, "reid_sha256": weight_hash, "versions": versions}
        csv_stream = open(run_dir / "tracks.csv", "w", encoding="utf-8-sig", newline="")
        csv_writer = csv.writer(csv_stream)
        csv_writer.writerow(["output_frame", "output_time_s", "camera", "source_frame", "source_time_s",
                             "global_id", "local_id", "segment_epoch", "x1", "y1", "x2", "y2",
                             "world_x_m", "world_y_m", "det_conf", "appearance_quality", "position_quality",
                             "status", "lookahead_s", "tail", "fresh_source",
                             "track_x1", "track_y1", "track_x2", "track_y2",
                             "detection_present", "occluded", "partial", "overlap_ratio",
                             "appearance_update_ok", "position_usable"])
        event_stream = open(run_dir / "events.jsonl", "w", encoding="utf-8")
        diagnostic_stream = open(run_dir / "diagnostics.jsonl", "w", encoding="utf-8")

        def event_sink(row):
            event_stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")

        manager, buffer = GlobalTracker(cfg, event_sink), FixedLagBuffer(cfg.delay_seconds, reader.fps)
        guard, ticks, previous = SegmentGuard(cfg), Counter(), {}
        processing_start = time.perf_counter()
        print(f"[시작] bbox={args.model}, imgsz={args.imgsz}, 지연={cfg.delay_seconds:.3f}s, "
              f"공통 입력={reader.fps:.3f} FPS, 보정={correction}, offset={args.cam2_offset_seconds:.3f}s", flush=True)
        print("[안내] 이미 출력한 ID는 소급 변경하지 않습니다. 출력 FPS 달성 여부는 실행 결과를 확인하세요.", flush=True)

        def emit(result):
            nonlocal rendered, tail_frames, maximum_age, age_sum, maximum_backlog
            nonlocal observed_total, identified_total
            packet, labels, lookahead, truncated = result
            tail_frames += int(truncated)
            with timings.measure("csv_output"):
                for cam, rows in labels.items():
                    for obs, gid, label in rows:
                        world = obs.world if obs.world is not None else ("", "")
                        raw_box = obs.bbox if obs.bbox is not None else ("",) * 4
                        track_box = obs.tracking_bbox if obs.tracking_bbox is not None else ("",) * 4
                        csv_writer.writerow([packet.sequence, packet.timestamp, cam, obs.frame, obs.timestamp,
                            gid, obs.local_id, obs.epoch, *raw_box, *world,
                            obs.confidence if obs.bbox is not None else "", obs.quality,
                            obs.position_quality, label, lookahead, int(truncated), int(packet.fresh[cam]),
                            *track_box, int(obs.bbox is not None), int(obs.occluded), int(obs.partial),
                            obs.overlap, int(obs.appearance_update_ok), int(position_reliable(obs, cfg))])
                    if packet.fresh[cam]:
                        observed_total += len(packet.observations.get(cam, []))
                        identified_total += len(rows)
                        diagnostic = dict(packet.diagnostics.get(cam, {}))
                        diagnostic.update(camera=cam, source_frame=packet.indices[cam],
                                          source_time_s=packet.times[cam], displayed=len(rows))
                        diagnostic["global_decisions"] = [
                            {"local_id": o.local_id, "epoch": o.epoch, "bbox": o.bbox.tolist(),
                             "reason": o.decision, "details": o.match_details,
                             "position_anchor_ok": o.position_anchor_ok,
                             "position_anchor_reason": o.position_anchor_reason}
                            for o in packet.observations.get(cam, [])]
                        diagnostic_stream.write(json.dumps(diagnostic, allow_nan=False) + "\n")
            if not args.no_video or args.show_preview:
                with timings.measure("render"):
                    frames = {cam: render(packet.frames[cam], labels[cam], cam, packet.indices[cam], packet.times[cam])
                              for cam in (1, 2)}
                    combined = combine_frames([frames[1], frames[2]], args.display_width)
                    rate = processed / max(loop_compute, 1e-6)
                    text = f"compute {rate:.1f} pairs/s | lag {lookahead:.2f}s | {'tail' if truncated else 'fixed lag'}"
                    cv2.putText(combined, text, (12, combined.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX,
                                .55, (255, 255, 255), 2)
                if not args.no_video:
                    with timings.measure("video_encode"):
                        if "combined" not in writers:
                            writers["combined"] = make_writer(run_dir / "combined.mp4", reader.fps,
                                (combined.shape[1], combined.shape[0]))
                            if args.save_separate:
                                for cam in (1, 2):
                                    writers[cam] = make_writer(run_dir / f"cam{cam}.mp4", reader.fps, sizes[cam])
                        writers["combined"].write(combined)
                        if args.save_separate:
                            for cam in (1, 2):
                                writers[cam].write(frames[cam])
                if args.show_preview:
                    cv2.imshow("MTMCT delayed bbox", combined)
                    if cv2.waitKey(1) & 0xff == ord("q"):
                        raise KeyboardInterrupt
            rendered += 1
            age = time.perf_counter() - packet.arrival
            maximum_age = max(maximum_age, age)
            age_sum += age
            if args.pace_input:
                late = time.perf_counter() - processing_start - packet.sequence / reader.fps
                maximum_backlog = max(maximum_backlog, late)

        while True:
            if args.pace_input:
                due = processing_start + reader.sequence / reader.fps
                wait = due - time.perf_counter()
                if wait > 0:
                    with timings.measure("input_pacing"):
                        time.sleep(wait)
            cycle = time.perf_counter()
            with timings.measure("video_decode"):
                packet = reader.next()
            if packet is None:
                break
            maximum_pair_skew = max(maximum_pair_skew, abs(packet.times[1] - packet.times[2]))
            detections, count = detect_batch(model, encoder, packet, calibration, trackers, ticks, args, cfg, timings)
            crop_count += count
            with timings.measure("local_botsort"):
                for cam in (1, 2):
                    if packet.fresh[cam]:
                        rows = local_update(trackers[cam], detections[cam], packet.frames[cam])
                        guard.update(rows, packet.timestamp)
                        packet.diagnostics[cam]["local_observations"] = len(rows)
                        packet.diagnostics[cam]["local_confirmed"] = sum(o.local_confirmed for o in rows)
                        packet.diagnostics[cam]["candidates"] = [
                            {"bbox": o.bbox.tolist(), "confidence": o.confidence, "local_id": o.local_id,
                             "partial": bool(o.partial), "overlap": o.overlap,
                             "appearance_quality": o.quality, "position_quality": o.position_quality}
                            for o in detections[cam]]
                        previous[cam] = rows
                        ticks[cam] += 1
                    packet.observations[cam] = previous.get(cam, [])
            buffer.push(packet)
            processed += 1
            while buffer.ready():
                with timings.measure("global_assignment"):
                    result = buffer.pop(manager)
                emit(result)
            loop_compute += time.perf_counter() - cycle
            if processed % 100 == 0:
                print(f"[처리] {processed}쌍 / 출력 {rendered} / 처리 {processed / loop_compute:.2f}쌍/s "
                      f"/ 활성 GID {len(manager.identities)}", flush=True)
        flush_start = time.perf_counter()
        while buffer.packets:
            with timings.measure("global_assignment"):
                result = buffer.pop(manager, flush=True)
            emit(result)
        loop_compute += time.perf_counter() - flush_start
        if processed == 0:
            raise RuntimeError("공통 시간 구간에 처리 가능한 프레임이 없습니다.")
        if rendered != processed:
            raise RuntimeError("처리/출력 프레임 수 불일치")
        status = "completed"
    except KeyboardInterrupt:
        status, failure = "interrupted", "User interrupted; remaining delayed frames were not emitted"
    except BaseException as exc:
        failure = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        for writer in writers.values():
            writer.release()
        if csv_stream:
            csv_stream.close()
        if event_stream:
            event_stream.close()
        if diagnostic_stream:
            diagnostic_stream.close()
        if reader:
            reader.close()
        if args.show_preview:
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass
        summary = {"version": VERSION, "status": status, "failure": failure,
            "baseline_source_sha256": BASELINE_SOURCE_SHA256,
            "settings": asdict(cfg), "arguments": vars(args), "models": model_info,
            "analyzed_pairs": processed, "rendered_pairs": rendered, "tail_frames": tail_frames,
            "osnet_crops": crop_count, "wall_seconds": time.perf_counter() - started,
            "processing_seconds_excluding_input_pacing": loop_compute,
            "compute_pairs_per_second": processed / loop_compute if loop_compute else None,
            "input_fps": reader.fps if reader else None,
            "meets_input_rate_average": bool(reader and loop_compute and processed / loop_compute >= reader.fps),
            "buffer_peak_pairs": buffer.peak if buffer else 0,
            "mean_decode_to_output_wall_seconds": age_sum / rendered if rendered else None,
            "max_decode_to_output_wall_seconds": maximum_age,
            "max_paced_source_to_output_wall_seconds": maximum_backlog if args.pace_input else None,
            "max_timestamp_residual_seconds": maximum_pair_skew,
            "timestamp_warning": "Residual uses configured alignment; it cannot measure unknown physical camera lag",
            "end_reason": reader.reason if reader else None,
            "source_repeated_frames": dict(reader.reused) if reader else {},
            "source_skipped_for_resampling": dict(reader.skipped) if reader else {},
            "video_info": reader.info if reader else {}, "correction": correction,
            "allocated_global_ids": manager.next_gid - 1 if manager else 0,
            "local_observations_total": observed_total,
            "identified_observations_total": identified_total,
            "unidentified_observation_ratio": ((observed_total-identified_total)/observed_total
                                                if observed_total else None),
            "retained_global_ids": sorted(manager.identities) if manager else [],
            "archived_global_ids": sorted(gid for gid, item in manager.identities.items()
                                           if item.archived_at is not None) if manager else [],
            "identity_retention_clock": "source_video_seconds_since_last_committed_match",
            "identity_retention_min_seconds": 300.,
            "events": dict(manager.events) if manager else {}, "timings": timings.report(),
            "deferred_by_reason": dict(manager.defer_counts) if manager else {},
            "duplicate_detections_removed": {cam: tr.duplicate_detections_removed for cam, tr in trackers.items()},
            "accuracy_evaluation": "NOT_MEASURED_NO_GROUND_TRUTH"}
        write_json(run_dir / "summary.json", summary)
        print(f"[종료] {status}: 분석 {processed}, 출력 {rendered}; {run_dir.resolve()}", flush=True)
        if processed and loop_compute:
            print(f"[속도] {processed / loop_compute:.2f} frame-pairs/s (decode+추론+할당+출력, pacing 제외)")
    return 0 if status == "completed" else 130


def self_test(include_botsort=False):
    """Synthetic regression tests only; they do not establish real-video accuracy."""
    import unittest

    class Tests(unittest.TestCase):
        def cross_owner_case(self, owner_cam=2):
            manager = GlobalTracker(self.association_cfg())
            manager.next_gid = 5
            fa = np.array([1., 0., 0., 0.], np.float32)
            fb = unit(np.array([.773, math.sqrt(1-.773**2), 0., 0.]))
            for gid, lid, x, feature in ((1, 36, 7., fa), (4, 30, 4.5, fb)):
                key = (owner_cam, lid, 0)
                identity = Identity(gid, 352/30)
                identity.profiles = {1: feature.copy(), 2: feature.copy()}
                identity.positions[owner_cam] = (np.array([x, 2.]), 352/30, .9)
                identity.keys[owner_cam] = key
                manager.identities[gid] = identity
                manager.bindings[key] = gid
                manager.binding_since[key] = (gid, 340/30)
            def rows(frame):
                a = self.obs(owner_cam, frame, lid=36, x=7.)
                b = self.obs(owner_cam, frame, lid=30, x=4.5)
                a.feature, b.feature = fa.copy(), fb.copy()
                return [a, b]
            for frame in range(352, 361):
                packet = self.packet(frame, rows(frame))
                manager.resolve(packet, [packet])
            cam = 3-owner_cam
            # Reproduce stale/wrong camera-specific anchors from the reported case.
            manager.identities[1].positions[cam] = (np.array([2.36, 2.]), 311/30, .9)
            manager.identities[4].positions[cam] = (np.array([6.60, 2.]), 353/30, .9)
            window = []
            for frame in range(361, 365):
                incoming = self.obs(cam, frame, lid=41, x=4.86)
                incoming.feature = fb.copy()
                window.append(self.packet(frame, rows(frame)+[incoming]))
            return manager, window, cam

        def test_v422_current_owner_prevents_far_id_theft_both_camera_orders(self):
            for owner_cam in (1, 2):
                manager, window, cam = self.cross_owner_case(owner_cam)
                output = manager.resolve(window[0], window)
                assigned = {obs.local_id: gid for rows in output.values() for obs, gid, _ in rows}
                self.assertEqual(assigned[36], 1)
                self.assertEqual(assigned[30], 4)
                self.assertEqual(assigned[41], 4)
                self.assertEqual(manager.next_gid, 5)
                incoming = next(r for r in window[0].observations[cam] if r.local_id == 41)
                audit = {}
                rejected = manager.identity_cost([describe(incoming, window, manager.cfg)],
                                                 manager.identities[1], 361/30, audit)
                self.assertTrue(math.isinf(rejected))
                self.assertEqual(audit['reason'], 'current_owner_position_conflict')

        def test_v422_pair_repairs_contaminated_anchor_only_at_source_time(self):
            manager, window, cam = self.cross_owner_case()
            manager.resolve(window[0], window)
            self.assertEqual(manager.events['cross_view_anchor_reset'], 1)
            position = manager.identities[4].positions[cam]
            np.testing.assert_allclose(position[0], [4.86, 2.])
            self.assertEqual(position[1], 361/30)
            self.assertEqual(manager.identities[4].last, 361/30)
            self.assertEqual(manager.cfg.identity_memory_seconds, 300.)

        def test_v422_one_pair_sample_cannot_reset_anchor(self):
            manager, window, cam = self.cross_owner_case()
            original = manager.identities[4].positions[cam][0].copy()
            manager.resolve(window[0], [window[0]])
            self.assertEqual(manager.events['cross_view_anchor_reset'], 0)
            np.testing.assert_array_equal(manager.identities[4].positions[cam][0], original)

        def test_v422_diverging_motion_cannot_reset_anchor(self):
            manager, window, cam = self.cross_owner_case()
            # Every instantaneous distance fits, but the two paths move oppositely.
            for i, packet in enumerate(window):
                next(o for o in packet.observations[cam] if o.local_id == 41).world[0] += .1*i
                next(o for o in packet.observations[3-cam] if o.local_id == 30).world[0] -= .1*i
            manager.resolve(window[0], window)
            self.assertEqual(manager.events['cross_view_anchor_reset'], 0)

        def test_v422_low_quality_floor_checks_conflict_without_anchor_write(self):
            manager = GlobalTracker(self.association_cfg())
            a, b = self.obs(1, 406, x=.81), self.obs(2, 406, x=6.45)
            b.position_quality = .378
            manager.record_motion(a)
            manager.record_motion(b)
            self.assertFalse(manager.anchor_observation_ok(b))
            self.assertIsNotNone(manager.position_conflict(a, b))
            manager.bindings = {a.key: 1, b.key: 1}
            self.assertTrue(manager.cross_conflict(describe(a, [], manager.cfg),
                                                   describe(b, [], manager.cfg)))

        def test_v422_bbox_jump_uses_recent_connected_floor_not_bad_new_foot(self):
            manager = GlobalTracker(self.association_cfg())
            old = self.obs(1, 379, x=2.7)
            manager.record_motion(old)
            a, b = self.obs(1, 380, x=6.9), self.obs(2, 380, x=7.)
            # Same image center, abrupt height change, displaced unreliable footpoint.
            a.bbox = old.bbox.copy()
            a.bbox[3] += 100
            manager.record_motion(a)
            self.assertEqual(a.position_anchor_reason, 'bbox_size_jump')
            self.assertFalse(manager.anchor_observation_ok(a))
            self.assertIsNotNone(manager.position_conflict(a, b))
            a.timestamp = 381/30 + manager.cfg.conflict_history_seconds
            a.world = None
            self.assertIsNone(manager.conflict_point(a))

        def test_v422_missing_floor_does_not_invent_conflict(self):
            manager = GlobalTracker(self.association_cfg())
            a, b = self.obs(1, 0), self.obs(2, 0, x=7.)
            a.world = None
            a.occluded = True
            self.assertIsNone(manager.position_conflict(a, b))

        def test_v422_existing_wrong_link_yields_to_older_owner(self):
            manager, window, cam = self.cross_owner_case()
            key = (cam, 41, 0)
            identity = manager.identities[1]
            identity.keys[cam] = key
            manager.bindings[key] = 1
            manager.binding_since[key] = (1, 354/30)
            for frame in range(354, 361):
                row = self.obs(cam, frame, lid=41, x=2.)
                manager.record_motion(row)
                manager.motion_history[key][-1]['reference_gid'] = 1
            identity.positions[cam] = (np.array([2., 2.]), 360/30, .9)
            rows = [self.obs(cam, 361, lid=41, x=2.), self.obs(3-cam, 361, lid=36, x=7.)]
            packet = self.packet(361, rows)
            output = manager.resolve(packet, [packet])
            self.assertEqual([(o.local_id, gid) for o, gid, _ in output[3-cam]], [(36, 1)])
            self.assertFalse(any(gid == 1 for _, gid, _ in output[cam]))
            self.assertIn(1, manager.identities)

        def motion_case(self, frames=7, missing_features=False, teleport=False):
            manager = GlobalTracker(self.association_cfg())
            first = self.obs(1, 0, x=1.)
            p = self.packet(0, [first])
            manager.resolve(p, [p])
            # Reproduce an already contaminated global anchor. The independent
            # local measurements remain close to the actual visible trajectory.
            manager.identities[1].positions[1] = (np.array([5., 2.]), 0., .9)
            results = []
            for f in range(1, frames):
                row = self.obs(1, f, x=1.+.03*f)
                if teleport:
                    row.bbox += np.array([600., 0., 600., 0.])
                if missing_features or f % 2:
                    row.feature = None
                    row.track_feature = np.eye(4, dtype=np.float32)[0]
                    row.track_feature_time = 0.
                p = self.packet(f, [row])
                out = manager.resolve(p, [p])
                results.append((row, out))
            return manager, results

        def test_v32_contaminated_anchor_recovers_from_independent_observations(self):
            manager, results = self.motion_case()
            self.assertFalse(results[0][1][1])
            self.assertEqual(results[1][1][1][0][1], 1)
            self.assertEqual(manager.events['motion_anchor_reset'], 1)
            self.assertEqual(manager.next_gid, 2)
            self.assertTrue(results[1][0].match_details['motion_anchor_reset'])
            np.testing.assert_allclose(manager.identities[1].positions[1][0], results[-1][0].world)

        def test_v32_unmatched_observations_do_not_refresh_identity(self):
            manager, results = self.motion_case(frames=2)
            self.assertFalse(results[0][1][1])
            self.assertEqual(len(manager.motion_history[results[0][0].key]), 2)
            self.assertEqual(manager.identities[1].last, 0.)
            np.testing.assert_array_equal(manager.identities[1].positions[1][0], [5., 2.])

        def test_v32_image_teleport_cannot_reanchor_same_appearance(self):
            manager, results = self.motion_case(teleport=True)
            self.assertTrue(all(not out[1] for _, out in results))
            self.assertEqual(manager.events['motion_anchor_reset'], 0)
            self.assertEqual(manager.identities[1].last, 0.)

        def test_v32_cached_features_cannot_vote_for_motion_recovery(self):
            manager, results = self.motion_case(missing_features=True)
            self.assertTrue(all(not out[1] for _, out in results))
            self.assertEqual(manager.events['motion_anchor_reset'], 0)

        def test_v32_occluded_footpoint_does_not_overwrite_trusted_anchor(self):
            manager = GlobalTracker(self.association_cfg())
            first = self.packet(0, [self.obs(1, 0)])
            manager.resolve(first, [first])
            original = manager.identities[1].positions[1][0].copy()
            row = self.obs(1, 1)
            row.occluded, row.position_usable = True, True
            row.continuity_ok, row.reliable_age = True, 1/30
            row.position_quality = .22
            row.world = np.array([5.7, -.62])  # Logged short-box failure pattern.
            p = self.packet(1, [row])
            result = manager.resolve(p, [p])
            self.assertEqual(result[1][0][1], 1)
            self.assertFalse(row.position_anchor_ok)
            np.testing.assert_array_equal(manager.identities[1].positions[1][0], original)

        def test_v32_full_body_return_is_not_a_new_immediate_anchor(self):
            manager = GlobalTracker(self.association_cfg())
            short = self.obs(1, 0)
            short.bbox[3] = short.bbox[1]+100
            short.partial, short.occluded = True, True
            manager.record_motion(short)
            full = self.obs(1, 1)
            manager.record_motion(full)
            self.assertEqual(full.position_anchor_reason, 'bbox_size_jump')
            self.assertFalse(manager.anchor_observation_ok(full))
            next_full = self.obs(1, 2)
            manager.record_motion(next_full)
            self.assertTrue(manager.anchor_observation_ok(next_full))

        def test_v32_fast_continuous_person_recovers_without_global_speed_relaxation(self):
            manager = GlobalTracker(self.association_cfg())
            p = self.packet(0, [self.obs(1, 0)])
            manager.resolve(p, [p])
            for f in range(1, 22):
                row = self.obs(1, f, x=1.+.20*f)
                if f < 15:
                    row.occluded = row.partial = row.continuity_ok = True
                    row.reliable_age = f/30
                    row.feature = row.world = None
                    row.position_quality = 0.
                elif f % 2:
                    row.feature = None
                p = self.packet(f, [row])
                result = manager.resolve(p, [p])
                if f >= 18:
                    self.assertEqual(result[1][0][1], 1)
            self.assertEqual(manager.events['motion_anchor_reset'], 1)
            self.assertEqual(manager.cfg.max_speed, 3.)
            self.assertEqual(manager.next_gid, 2)

        def test_v32_current_cross_camera_conflict_vetoes_recovery_before_assignment(self):
            manager, _ = self.motion_case(frames=2)
            other = self.obs(2, 2, x=4.)
            manager.bindings[other.key] = 1
            identity = manager.identities[1]
            identity.keys[2] = other.key
            identity.profiles[2] = other.feature.copy()
            identity.positions[2] = (other.world.copy(), 0., .9)
            for f in range(2, 6):
                a = self.obs(1, f, x=1.+.03*f)
                b = self.obs(2, f, x=4.)
                p = self.packet(f, [a, b])
                result = manager.resolve(p, [p])
                self.assertFalse(result[1])
                self.assertEqual(result[2][0][1], 1)
            self.assertEqual(manager.events['motion_anchor_reset'], 0)

        def test_v32_repeated_frames_and_future_frames_cannot_supply_recovery_votes(self):
            manager, results = self.motion_case(frames=2)
            row = results[0][0]
            for _ in range(10):
                manager.record_motion(row)
            self.assertEqual(len(manager.motion_history[row.key]), 2)
            future = [self.packet(f, [self.obs(1, f, x=1.+.03*f)]) for f in range(2, 10)]
            d = describe(row, [self.packet(1, [row])]+future, manager.cfg)
            self.assertIsNone(manager.recover_motion(d, manager.identities[1]))

        def test_v31_soft_margin_only_rewards_available_old_id(self):
            manager = GlobalTracker(Config())
            row = self.obs(1, 1)
            manager.bindings[row.key] = 1
            units = [[Descriptor(row, [row], row.feature, None)]]
            raw = np.array([[.20, .18]])
            scored = manager.prefer_valid_continuation(raw, units, [1, 2])
            self.assertEqual(assign(scored, .64), [(0, 0)])
            np.testing.assert_allclose(raw, [[.20, .18]])
            # Strong alternatives are still usable immediately, even while the
            # old ID is valid. An unavailable old ID receives no extra hold.
            for costs in ([[.20, .05]], [[math.inf, .18]], [[.80, .18]]):
                scored = manager.prefer_valid_continuation(np.array(costs), units, [1, 2])
                self.assertEqual(assign(scored, .64), [(0, 1)])

        def test_v31_wrong_existing_id_corrects_without_wait(self):
            manager = GlobalTracker(self.association_cfg())
            initial = self.packet(0, [self.obs(1, 0, person=p) for p in (0, 1)])
            manager.resolve(initial, [initial])
            # The same Local ID now has unmistakable evidence for person 1.
            row = self.obs(1, 1, person=1, lid=1)
            packet = self.packet(1, [row])
            result = manager.resolve(packet, [packet])
            self.assertEqual(result[1][0][1], 2)
            self.assertEqual(manager.events['reassignment'], 1)
            self.assertEqual(manager.events['deferred_observations'], 0)

        def test_v31_fallback_solves_competition_without_forcing_ambiguity(self):
            matrix = np.array([[.10, .16], [.11, math.inf]])
            self.assertEqual(set(recovery_assignment(matrix, .64, .04, set())), {(0, 1), (1, 0)})
            self.assertEqual(recovery_assignment(np.full((2, 2), .10), .64, .04, set()), [])
            self.assertEqual(recovery_assignment([[.63]], .64, .04, set()), [])
            self.assertEqual(recovery_assignment([[math.inf]], .64, .04, set()), [])

        def test_v31_failed_pairs_batch_recovers_person_lost_by_sequential_v3(self):
            def scenario(batch):
                manager = GlobalTracker(self.association_cfg(batch_pair_fallback=batch, continuity_margin=0.))
                manager.identities = {g: Identity(g, 0.) for g in range(1, 7)}
                manager.next_gid = 7
                rows = [self.obs(cam, 1, lid=lid, x=float(lid)) for cam in (1, 2) for lid in (1, 2)]
                manager.bindings = {r.key: (r.local_id+2 if r.cam == 1 else r.local_id+4) for r in rows}
                manager.pair_cost = lambda a, b: 0. if a.first.local_id == b.first.local_id else math.inf
                # Isolate assignment logic from the encoder: A can use 1 or 2;
                # B can use only 1. Each proposed cross-camera group is invalid.
                def controlled_cost(group, identity, now, audit=None):
                    if audit is not None:
                        audit.update(reason='controlled_test_cost', gid=identity.gid)
                    if len(group) != 1:
                        return math.inf
                    d = group[0]
                    return {(1, 1, 1): .10, (1, 1, 2): .16, (1, 2, 1): .11,
                            (2, 1, 5): .05, (2, 2, 6): .05}.get(
                                (d.cam, d.first.local_id, identity.gid), math.inf)
                manager.identity_cost = controlled_cost
                packet = self.packet(1, rows)
                return manager.resolve(packet, [packet]), manager
            old, old_manager = scenario(False)
            new, new_manager = scenario(True)
            self.assertEqual(len(old[1]), 1)
            self.assertEqual({o.local_id: gid for o, gid, _ in new[1]}, {1: 2, 2: 1})
            self.assertEqual(new_manager.events['deferred_observations'], 0)
            self.assertEqual(old_manager.events['deferred_observations'], 1)
            self.assertEqual(new_manager.next_gid, 7)

        def test_v31_retention_300_source_seconds_and_match_refresh(self):
            with self.assertRaises(ValueError):
                Config(identity_memory_seconds=299.)
            manager = GlobalTracker(self.association_cfg())
            initial = self.packet(0, [self.obs(1, 0)])
            manager.resolve(initial, [initial])
            at_300 = self.packet(9000, [])
            manager.resolve(at_300, [at_300])
            self.assertIn(1, manager.identities)
            again = self.packet(9000, [self.obs(1, 9000)])
            self.assertEqual(manager.resolve(again, [again])[1][0][1], 1)
            self.assertEqual(manager.identities[1].last, 300.)
            at_600 = self.packet(18000, [])
            manager.resolve(at_600, [at_600])
            self.assertIn(1, manager.identities)
            expired = self.packet(18001, [])
            manager.resolve(expired, [expired])
            self.assertNotIn(1, manager.identities)
            self.assertEqual(manager.events['identity_expired'], 1)

        def test_v31_archive_does_not_refresh_expiry_and_can_restore(self):
            manager = GlobalTracker(self.association_cfg())
            row = self.obs(1, 1)
            manager.identities[1] = Identity(1, 0., profiles={1: row.feature.copy()},
                                             positions={1: (row.world.copy(), 0., .9)}, archived_at=0.)
            manager.next_gid = 2
            packet = self.packet(1, [row])
            result = manager.resolve(packet, [packet])
            self.assertEqual(result[1][0][1], 1)
            self.assertIsNone(manager.identities[1].archived_at)
            self.assertEqual(manager.events['identity_restored'], 1)
            manager.identities[2] = Identity(2, 0., archived_at=.01)
            just_before = self.packet(8999, [])
            manager.resolve(just_before, [just_before])
            self.assertEqual(manager.identities[2].last, 0.)
            expired = self.packet(9001, [])
            manager.resolve(expired, [expired])
            self.assertNotIn(2, manager.identities)

        def test_v31_archive_never_competes_with_occupied_live_duplicate(self):
            manager = GlobalTracker(self.association_cfg())
            a, b = self.obs(1, 1, lid=1), self.obs(1, 1, lid=2, x=1.2)
            for gid in (1, 2):
                manager.identities[gid] = Identity(gid, 0., profiles={1: a.feature.copy()},
                    positions={1: (a.world.copy(), 0., .9)}, archived_at=0. if gid == 1 else None)
            manager.bindings[a.key] = 2
            manager.next_gid = 3
            packet = self.packet(1, [a, b])
            result = manager.resolve(packet, [packet])
            self.assertNotIn(1, [g for _, g, _ in result[1]])
            self.assertIsNotNone(manager.identities[1].archived_at)
            self.assertEqual(manager.identities[1].last, 0.)

        def test_v31_diagnostics_are_finite_and_explain_rejections(self):
            manager = GlobalTracker(self.association_cfg())
            p = self.packet(0, [self.obs(1, 0)])
            manager.resolve(p, [p])
            row = self.obs(1, 1, x=30.)
            p = self.packet(1, [row])
            manager.resolve(p, [p])
            encoded = json.dumps(row.match_details, allow_nan=False)
            self.assertIn('motion_conflict', encoded)
            self.assertIn('previous_gid', encoded)

        def association_cfg(self, **kwargs):
            # Existing association tests isolate matching from confirmation delay.
            # Dedicated lifecycle tests below use the production Config defaults.
            return Config(birth_min_hits=1, birth_min_seconds=0., **kwargs)

        def test_v3_moderate_overlap_keeps_comparison_and_floor_evidence(self):
            cfg = Config()
            local = SimpleNamespace(tracked_stracks=[], lost_stracks=[])
            rows = [self.obs(1, 0, p) for p in (0, 1)]
            rows[0].bbox = np.array([100., 100., 200., 300.])
            rows[1].bbox = np.array([160., 100., 260., 300.])
            prepare_occlusions(local, rows, cfg)
            self.assertAlmostEqual(iou(rows[0].bbox, rows[1].bbox), .25)
            for row in rows:
                self.assertTrue(row.occluded)
                self.assertFalse(row.partial)
                self.assertIsNotNone(row.feature)
                self.assertGreaterEqual(row.quality, cfg.feature_quality)
                self.assertTrue(position_reliable(row, cfg))
                self.assertFalse(row.appearance_update_ok)
                self.assertTrue(local_candidate_ok(row, cfg))
                self.assertFalse(birth_observation_ok(row, cfg))

        def test_v3_fragment_does_not_mark_whole_person_fully_occluded(self):
            whole = np.array([100., 100., 200., 300.])
            fragment = np.array([120., 250., 180., 300.])
            ratios = overlap_ratios([whole, fragment])
            self.assertAlmostEqual(float(ratios[0]), .15)
            self.assertAlmostEqual(float(ratios[1]), 1.)

        def test_v3_overlap_can_establish_cross_camera_match(self):
            cfg, manager = Config(), GlobalTracker(Config())
            frames = []
            for f in range(4):
                a, b = self.obs(1, f), self.obs(2, f)
                a.occluded, a.overlap, a.appearance_update_ok = True, .4, False
                a.position_usable = True
                frames.append(self.packet(f, [a, b]))
            a = describe(frames[0].observations[1][0], frames, cfg)
            b = describe(frames[0].observations[2][0], frames, cfg)
            self.assertLess(manager.pair_cost(a, b), cfg.pair_cost_limit)
            # A partial body without trustworthy floor observations still cannot
            # use appearance alone to establish a new cross-camera match.
            for packet in frames:
                row = packet.observations[1][0]
                row.partial, row.position_usable = True, False
            a = describe(frames[0].observations[1][0], frames, cfg)
            self.assertFalse(np.isfinite(manager.pair_cost(a, b)))

        def test_v3_existing_identity_recovered_before_birth_confirmation(self):
            manager = GlobalTracker(Config())
            for f in range(4):
                packet = self.packet(f, [self.obs(1, f)])
                manager.resolve(packet, [packet])
            row = self.obs(1, 4, lid=99)
            packet = self.packet(4, [row])
            labels = manager.resolve(packet, [packet])
            self.assertEqual(labels[1][0][1], 1)
            self.assertFalse(manager.birth_ready(describe(row, [packet], manager.cfg)))
            self.assertEqual(manager.next_gid, 2)
            self.assertEqual(manager.events["identity_recovered"], 1)

        def test_v3_dormant_recovery_uses_appearance_not_stale_coordinates(self):
            manager = GlobalTracker(Config())
            for f in range(4):
                packet = self.packet(f, [self.obs(1, f)])
                manager.resolve(packet, [packet])
            empty = self.packet(150, [])
            manager.resolve(empty, [empty])
            self.assertIn(1, manager.identities)
            self.assertAlmostEqual(manager.identities[1].last, .1)
            row = self.obs(1, 151, lid=99, x=20.)
            packet = self.packet(151, [row])
            labels = manager.resolve(packet, [packet])
            self.assertEqual(labels[1][0][1], 1)
            self.assertEqual(manager.next_gid, 2)

        def test_v3_ambiguous_dormant_appearances_are_not_arbitrarily_recovered(self):
            manager = GlobalTracker(Config())
            for gid in (1, 2):
                identity = Identity(gid, 0.)
                identity.profiles[1] = self.obs(1, 0).feature.copy()
                identity.positions[1] = (np.array([float(gid), 2.]), 0., .9)
                manager.identities[gid] = identity
            manager.next_gid = 3
            for f in range(150, 157):
                packet = self.packet(f, [self.obs(1, f, lid=99)])
                self.assertFalse(manager.resolve(packet, [packet])[1])
            self.assertEqual(manager.next_gid, 3)
            self.assertGreater(manager.defer_counts["ambiguous_existing_identity"], 0)

        def test_v3_low_similarity_cannot_recover_dormant_identity(self):
            manager = GlobalTracker(Config())
            identity = Identity(1, 0.)
            identity.profiles[1] = self.obs(1, 0).feature.copy()
            manager.identities[1], manager.next_gid = identity, 2
            row = self.obs(1, 150, lid=99)
            row.feature = np.array([.60, .80, 0., 0.])
            packet = self.packet(150, [row])
            self.assertFalse(manager.resolve(packet, [packet])[1])
            self.assertEqual(manager.next_gid, 2)

        def test_v3_known_small_full_box_retains_evidence(self):
            cfg = Config()
            box = np.array([100., 100., 128., 210.])
            track = SimpleNamespace(track_id=1, reliable_box=box.copy(), reliable_time=0.,
                                    box_velocity=np.zeros(4))
            local = SimpleNamespace(tracked_stracks=[track], lost_stracks=[])
            row = self.obs(1, 1)
            row.bbox = box.copy()
            prepare_occlusions(local, [row], cfg)
            self.assertFalse(row.partial)
            self.assertIsNotNone(row.feature)
            self.assertTrue(position_reliable(row, cfg))
            self.assertFalse(birth_observation_ok(row, cfg))

        def test_v3_history_snapshot_expires_without_self_refresh(self):
            cfg = Config()
            for f, expected in ((10, True), (40, False)):
                row = self.obs(1, f)
                row.feature = None
                row.track_feature = np.array([1., 0., 0., 0.])
                row.track_feature_time = 0.
                packet = self.packet(f, [row])
                descriptor = describe(row, [packet], cfg)
                self.assertEqual(descriptor.feature is not None, expected)
                if expected:
                    self.assertTrue(descriptor.feature_from_history)

        def test_v3_measured_overlap_compares_without_updating_trusted_profile(self):
            cfg, manager = self.association_cfg(), GlobalTracker(self.association_cfg())
            packet = self.packet(0, [self.obs(1, 0)])
            manager.resolve(packet, [packet])
            before = manager.identities[1].profiles[1].copy()
            row = self.obs(1, 1)
            row.occluded, row.position_usable, row.appearance_update_ok = True, True, False
            row.feature = np.array([.9, np.sqrt(1-.9**2), 0., 0.])
            packet = self.packet(1, [row])
            self.assertEqual(manager.resolve(packet, [packet])[1][0][1], 1)
            np.testing.assert_array_equal(manager.identities[1].profiles[1], before)

        def test_v3_real_local_overlap_candidate_can_recover_existing_global(self):
            if not include_botsort:
                self.skipTest("--test-botsort requires torch/ultralytics")
            cfg, manager = Config(), GlobalTracker(Config())
            for f in range(4):
                packet = self.packet(f, [self.obs(1, f)])
                manager.resolve(packet, [packet])
            local = make_local_tracker(30, cfg)
            image = np.zeros((720, 1280, 3), np.uint8)
            recovered = []
            for f in range(4, 7):
                row = self.obs(1, f)
                row.occluded, row.overlap = True, .4
                row.position_usable, row.appearance_update_ok = True, False
                out = local_update(local, [row], image)
                self.assertTrue(out)
                # Simulate a local restart with a distinct stable key.
                for r in out:
                    r.local_id += 100
                packet = self.packet(f, out)
                recovered.extend(g for _, g, _ in manager.resolve(packet, [packet])[1])
            self.assertTrue(recovered)
            self.assertEqual(set(recovered), {1})
            self.assertEqual(manager.next_gid, 2)

        def test_v3_superseded_identity_does_not_remain_as_duplicate_memory(self):
            cfg, manager = self.association_cfg(), GlobalTracker(self.association_cfg())
            first = self.packet(0, [self.obs(1, 0), self.obs(2, 0, x=6.)])
            manager.resolve(first, [first])
            self.assertEqual(len(manager.identities), 2)
            window = [self.packet(f, [self.obs(1, f), self.obs(2, f)]) for f in range(1, 5)]
            labels = manager.resolve(window[0], window)
            self.assertEqual(labels[1][0][1], labels[2][0][1])
            self.assertEqual(len(manager.identities), 2)  # User's 300-second retention rule.
            self.assertEqual(sum(x.archived_at is not None for x in manager.identities.values()), 1)
            winner = labels[1][0][1]
            later = self.packet(5, [self.obs(1, 5, lid=50)])
            recovered = manager.resolve(later, [later])
            self.assertEqual(recovered[1][0][1], winner)
            self.assertEqual(manager.next_gid, 3)

        def test_v3_segment_epoch_survives_long_occlusion(self):
            guard = SegmentGuard(Config())
            a, b = self.obs(1, 0), self.obs(1, 1, person=1, lid=1)
            guard.update([a], a.timestamp)
            guard.update([b], b.timestamp)
            self.assertEqual(b.epoch, 1)
            hidden = self.obs(1, 150, person=1, lid=1)
            hidden.occluded, hidden.partial = True, True
            guard.update([hidden], hidden.timestamp)
            self.assertEqual(hidden.epoch, 1)

        def test_production_birth_waits_for_time_and_distinct_observations(self):
            manager = GlobalTracker(Config())
            for f in range(8):
                packet = self.packet(f, [self.obs(1, f)])
                # Rich lookahead must not publish an ID ahead of the confirmation time.
                window = [self.packet(k, [self.obs(1, k)]) for k in range(f, f + 10)]
                labels = manager.resolve(packet, window)
                if f < 3:
                    self.assertFalse(labels[1])
                    self.assertEqual(manager.next_gid, 1)
                else:
                    self.assertEqual(labels[1][0][1], 1)
            self.assertEqual(manager.next_gid, 2)

        def test_production_single_frame_candidate_gets_no_id_even_at_eof(self):
            manager, buffer = GlobalTracker(Config()), FixedLagBuffer(.3, 30)
            for f in range(5):
                buffer.push(self.packet(f, [self.obs(1, f)] if f == 4 else []))
            while buffer.packets:
                result = buffer.pop(manager, flush=True)
                self.assertEqual(result[1], {1: [], 2: []})
            self.assertEqual(manager.next_gid, 1)

        def test_production_repeated_frame_and_unconfirmed_tracks_do_not_confirm(self):
            manager = GlobalTracker(Config())
            for f in range(10):
                row = self.obs(1, f)
                row.frame = 0
                packet = self.packet(f, [row])
                self.assertFalse(manager.resolve(packet, [packet])[1])
            self.assertEqual(manager.next_gid, 1)
            manager = GlobalTracker(Config())
            for f in range(10):
                row = self.obs(1, f)
                row.local_confirmed = False
                packet = self.packet(f, [row])
                self.assertFalse(manager.resolve(packet, [packet])[1])
            self.assertEqual(manager.next_gid, 1)

        def test_production_uncertain_occlusion_holds_binding_without_new_id(self):
            manager = GlobalTracker(Config())
            for f in range(14):
                row = self.obs(1, f)
                if 5 <= f < 10:
                    row.occluded = row.partial = True
                    row.continuity_ok = False
                    row.world = row.feature = None
                    row.quality = row.position_quality = 0.
                packet = self.packet(f, [row])
                labels = manager.resolve(packet, [packet])
                if 5 <= f < 10:
                    self.assertFalse(labels[1])
                    self.assertEqual(manager.bindings[row.key], 1)
                    self.assertAlmostEqual(manager.identities[1].last, 4 / 30)
                elif f >= 3:
                    self.assertEqual(labels[1][0][1], 1)
            self.assertEqual(manager.next_gid, 2)

        def test_production_existing_match_rejection_does_not_issue_replacement(self):
            manager = GlobalTracker(Config())
            for f in range(10):
                row = self.obs(1, f, person=1 if 5 <= f < 8 else 0, lid=1)
                packet = self.packet(f, [row])
                labels = manager.resolve(packet, [packet])
                if 5 <= f < 8:
                    self.assertFalse(labels[1])
                elif f >= 3:
                    self.assertEqual(labels[1][0][1], 1)
            self.assertEqual(manager.next_gid, 2)

        def test_production_box_floor_scales_with_resolution(self):
            row = self.obs(1, 0)
            row.bbox = np.array([10., 10., 22., 50.])
            row.image_size = (640, 360)
            self.assertTrue(bbox_large_enough(row, 24, 80))
            self.assertFalse(birth_observation_ok(row, Config()))
            row.image_size = (1280, 720)
            self.assertFalse(bbox_large_enough(row, 24, 80))

        def test_production_partial_and_low_confidence_never_start_local_tracks(self):
            if not include_botsort:
                self.skipTest("--test-botsort requires torch/ultralytics")
            cfg = Config()
            image = np.zeros((720, 1280, 3), np.uint8)
            for kind in ("fragment", "low_confidence", "too_small"):
                tracker = make_local_tracker(30, cfg)
                for f in range(10):
                    row = self.obs(1, f)
                    if kind == "fragment":
                        row.bbox[3] = row.bbox[1] + 100
                    elif kind == "too_small":
                        row.bbox[3] = row.bbox[1] + 45
                    else:
                        row.confidence = .49
                    prepare_occlusions(tracker, [row], cfg)
                    self.assertFalse(local_update(tracker, [row], image), kind)
                self.assertFalse(tracker.tracked_stracks, kind)

        def test_production_duplicate_iou_085_is_effective(self):
            tracker = SimpleNamespace(tracked_stracks=[], lost_stracks=[], duplicate_detections_removed=0)
            boxes = np.array([[100, 100, 150, 300], [103, 100, 153, 300]], np.float32)
            self.assertGreater(iou(boxes[0], boxes[1]), .85)
            self.assertLess(iou(boxes[0], boxes[1]), .95)
            keep = deduplicate_detections(boxes, [.9, .7], tracker, 0., Config())
            self.assertEqual(keep.tolist(), [0])

        def obs(self, cam, frame, person=0, lid=None, x=None, epoch=0):
            feature = np.eye(4, dtype=np.float32)[person]
            x = 1. + person * 2 if x is None else x
            return Observation(cam, frame, frame / 30, np.array([100 + x * 30, 100, 150 + x * 30, 300.]),
                               .9, .9, .9, np.array([x, 2.]), feature=feature,
                               local_id=person + 1 if lid is None else lid, epoch=epoch)

        def packet(self, frame, rows):
            return Packet(frame, frame / 30, {}, {1: frame, 2: frame}, {1: frame / 30, 2: frame / 30},
                          {1: True, 2: True}, {cam: [d for d in rows if d.cam == cam] for cam in (1, 2)})

        def run_rows(self, frames):
            tracker, buffer = GlobalTracker(self.association_cfg()), FixedLagBuffer(.3, 30)
            out = []
            for f, rows in enumerate(frames):
                buffer.push(self.packet(f, rows))
                while buffer.ready():
                    out.append(buffer.pop(tracker))
            while buffer.packets:
                out.append(buffer.pop(tracker, flush=True))
            return out, tracker, buffer

        def test_assignment_global_optimum(self):
            self.assertEqual(set(assign([[2, 3], [3, 100]], 200)), {(0, 1), (1, 0)})

        def test_assignment_unmatched(self):
            self.assertEqual(assign([[math.inf, .2], [math.inf, math.inf]], .6), [(0, 1)])
            self.assertEqual(assign(np.empty((0, 2)), .6), [])

        def test_pairing_and_immutable_output(self):
            frames = [[self.obs(c, f, p) for c in (1, 2) for p in (0, 1)] for f in range(45)]
            out, tracker, buffer = self.run_rows(frames)
            self.assertEqual(len(out), 45)
            ids = {p: set() for p in (0, 1)}
            for packet, labels, lag, tail in out:
                self.assertLessEqual(lag, .3 + 1e-8)
                for rows in labels.values():
                    self.assertEqual(len({g for _, g, _ in rows}), len(rows))
                    for obs, gid, status in rows:
                        ids[int(np.argmax(obs.feature))].add(gid)
            self.assertEqual([len(ids[p]) for p in (0, 1)], [1, 1])
            self.assertNotEqual(ids[0], ids[1])
            self.assertEqual(tracker.next_gid - 1, 2)
            self.assertLessEqual(buffer.peak, 11)

        def test_delay_is_real_lookahead(self):
            tracker, buffer = GlobalTracker(self.association_cfg()), FixedLagBuffer(.3, 30)
            for f in range(9):
                buffer.push(self.packet(f, [self.obs(c, f) for c in (1, 2)]))
                self.assertFalse(buffer.ready())
            buffer.push(self.packet(9, [self.obs(c, 9) for c in (1, 2)]))
            packet, labels, lag, tail = buffer.pop(tracker)
            self.assertEqual(packet.sequence, 0)
            self.assertAlmostEqual(lag, .3)
            self.assertFalse(tail)
            self.assertEqual(labels[1][0][1], labels[2][0][1])

        def test_local_id_switch_reassignment(self):
            frames = []
            for f in range(60):
                frames.append([self.obs(c, f, p, lid=(2 - p if c == 1 and f >= 20 else p + 1),
                                        epoch=int(c == 1 and f >= 20)) for c in (1, 2) for p in (0, 1)])
            out, _, _ = self.run_rows(frames)
            ids = {p: set() for p in (0, 1)}
            for _, labels, _, _ in out:
                for rows in labels.values():
                    for obs, gid, _ in rows:
                        ids[int(np.argmax(obs.feature))].add(gid)
            self.assertEqual([len(ids[0]), len(ids[1])], [1, 1])

        def test_occlusion_new_local_id(self):
            frames = []
            for f in range(60):
                rows = [self.obs(2, f)]
                if not 15 <= f < 25:
                    rows.append(self.obs(1, f, lid=1 if f < 15 else 9))
                frames.append(rows)
            out, tracker, _ = self.run_rows(frames)
            ids = {gid for _, labels, _, _ in out for rows in labels.values() for _, gid, _ in rows}
            self.assertEqual(len(ids), 1)
            self.assertEqual(tracker.next_gid - 1, 1)

        def test_far_same_appearance_not_merged(self):
            frames = [[self.obs(1, f), self.obs(2, f, x=6.)] for f in range(20)]
            out, _, _ = self.run_rows(frames)
            for _, labels, _, _ in out:
                self.assertNotEqual(labels[1][0][1], labels[2][0][1])

        def test_segment_discontinuity(self):
            guard = SegmentGuard(self.association_cfg())
            first = self.obs(1, 0)
            guard.update([first], 0.)
            second = self.obs(1, 1)
            second.feature = np.array([.32, np.sqrt(1 - .32 ** 2), 0., 0.], np.float32)
            guard.update([second], 1 / 30)
            self.assertEqual(second.epoch, first.epoch + 1)

        def test_unreliable_position_no_cross_merge(self):
            frames = []
            for f in range(20):
                a, b = self.obs(1, f), self.obs(2, f)
                a.position_quality = b.position_quality = .01
                frames.append([a, b])
            out, _, _ = self.run_rows(frames)
            for _, labels, _, _ in out:
                self.assertNotEqual(labels[1][0][1], labels[2][0][1])

        def test_ambiguous_pairs_not_merged_through_singles(self):
            frames = [[self.obs(c, f, lid=lid) for c in (1, 2) for lid in (1, 2)] for f in range(20)]
            out, _, _ = self.run_rows(frames)
            for _, labels, _, _ in out:
                a = {gid for _, gid, _ in labels[1]}
                b = {gid for _, gid, _ in labels[2]}
                self.assertFalse(a & b)

        def test_repeated_source_keeps_published_id(self):
            cfg = self.association_cfg()
            tracker, buffer = GlobalTracker(cfg), FixedLagBuffer(.3, 30)
            results = []
            last_cam2 = None
            for f in range(25):
                if f != 12:
                    last_cam2 = self.obs(2, f if f < 12 else f - 1)
                    if f > 12:
                        last_cam2.timestamp = f / 30
                packet = self.packet(f, [self.obs(1, f), last_cam2])
                packet.indices[2] = last_cam2.frame
                packet.times[2] = last_cam2.timestamp
                packet.fresh[2] = f != 12
                buffer.push(packet)
                while buffer.ready():
                    results.append(buffer.pop(tracker))
            while buffer.packets:
                results.append(buffer.pop(tracker, flush=True))
            self.assertEqual(results[11][1][2][0][1], results[12][1][2][0][1])
            self.assertEqual(results[11][1][2][0][0].frame, results[12][1][2][0][0].frame)

        def make_video(self, path, fps, count):
            writer = make_writer(path, fps, (128, 72))
            try:
                for f in range(count):
                    writer.write(np.full((72, 128, 3), f * 3 % 255, np.uint8))
            finally:
                writer.release()

        def test_video_reader_unequal_fps(self):
            import tempfile
            with tempfile.TemporaryDirectory() as directory:
                paths = [Path(directory) / f"cam{cam}.mp4" for cam in (1, 2)]
                self.make_video(paths[0], 30, 30)
                self.make_video(paths[1], 15, 15)
                reader = VideoPairReader(paths, 0, None, ("none",))
                packets = []
                try:
                    while (packet := reader.next()) is not None:
                        packets.append(packet)
                finally:
                    reader.close()
                self.assertEqual(reader.fps, 15)
                self.assertEqual(len(packets), 15)
                self.assertEqual([p.indices[1] for p in packets], list(range(0, 30, 2)))
                self.assertTrue(all(abs(p.times[1] - p.times[2]) < 1e-8 for p in packets))

        def test_video_reader_step_correction_and_early_decode_error(self):
            import tempfile
            with tempfile.TemporaryDirectory() as directory:
                paths = [Path(directory) / f"cam{cam}.mp4" for cam in (1, 2)]
                for path in paths:
                    self.make_video(path, 30, 25)
                reader = VideoPairReader(paths, 0, None, ("step", 12, -1))
                try:
                    packets = [reader.next() for _ in range(25)]
                    self.assertFalse(packets[12].fresh[2])
                    self.assertEqual(packets[12].indices[2], 11)
                    self.assertIsNone(reader.next())
                finally:
                    reader.close()
                reader = VideoPairReader(paths, 0, None, ("none",))
                reader.caps[1].release()
                try:
                    with self.assertRaisesRegex(RuntimeError, "decode"):
                        reader.next()
                finally:
                    reader.close()

        def test_noninteger_delay_no_extra_future(self):
            tracker, buffer = GlobalTracker(self.association_cfg(delay_seconds=.31)), FixedLagBuffer(.31, 30)
            for f in range(11):
                buffer.push(self.packet(f, [self.obs(c, f) for c in (1, 2)]))
            result = buffer.pop(tracker)
            self.assertLessEqual(result[2], .31)

        def test_calibration_original_points(self):
            c = Calibration()
            for cam, src, dst in ((1, CAM1_IMAGE_POINTS, WORLD_POINTS_CAM1),
                                  (2, CAM2_IMAGE_POINTS, WORLD_POINTS_CAM2)):
                for pixel, world in zip(src, dst):
                    self.assertLess(np.linalg.norm(c.project(cam, pixel, (1280, 720)) - world), 1e-5)
                    self.assertLess(np.linalg.norm(c.project(cam, pixel / 2, (640, 360)) - world), 1e-5)

        def test_feature_boxes_original_indices(self):
            rows = [self.obs(1, 0, p) for p in range(3)]
            boxes = FeatureBoxes(rows)[np.array([False, True, True])][np.array([True, False])]
            self.assertEqual(boxes.indices.tolist(), [1])

        def test_short_clip_flush_and_no_boxes_in_gap(self):
            frames = [[self.obs(c, f) for c in (1, 2)] if f != 2 else [] for f in range(5)]
            out, _, _ = self.run_rows(frames)
            self.assertEqual(len(out), 5)
            self.assertEqual(out[2][1], {1: [], 2: []})
            self.assertTrue(all(row[3] for row in out))

        def test_real_botsort_missing_features_low_score(self):
            if not include_botsort:
                self.skipTest("--test-botsort requires torch/ultralytics")
            tracker = make_local_tracker(30, self.association_cfg())
            image = np.zeros((720, 1280, 3), np.uint8)
            ids = set()
            for f in range(40):
                rows = [] if 10 <= f < 13 else [self.obs(1, f, x=1 + .005 * f)]
                if rows and f % 2:
                    rows[0].feature = None
                if rows and 20 <= f < 23:
                    rows[0].confidence = .15
                out = local_update(tracker, rows, image)
                if rows:
                    self.assertEqual(len(out), 1)
                    ids.add(out[0].local_id)
            self.assertEqual(len(ids), 1)

        def test_occluded_reference_and_epoch_are_frozen(self):
            cfg, guard = self.association_cfg(), SegmentGuard(self.association_cfg())
            tracker = GlobalTracker(cfg)
            first = self.obs(1, 0)
            guard.update([first], 0.)
            packet = self.packet(0, [first])
            gid = tracker.resolve(packet, [packet])[1][0][1]
            original_profile = tracker.identities[gid].profiles[1].copy()
            original_position = tracker.identities[gid].positions[1][0].copy()
            for f in range(1, 15):
                row = self.obs(1, f, person=1, lid=1, x=99.)
                row.occluded, row.continuity_ok, row.reliable_age = True, True, f / 30
                row.partial = True  # Cropped, unreliable appearance and floor point.
                guard.update([row], f / 30)
                packet = self.packet(f, [row])
                labels = tracker.resolve(packet, [packet])
                self.assertEqual(labels[1][0][1], gid)
                self.assertEqual(row.epoch, 0)
            np.testing.assert_array_equal(tracker.identities[gid].profiles[1], original_profile)
            np.testing.assert_array_equal(tracker.identities[gid].positions[1][0], original_position)
            restored = self.obs(1, 15)
            guard.update([restored], .5)
            self.assertEqual(restored.epoch, 0)

        def test_occluded_new_camera_does_not_claim_existing_identity(self):
            tracker = GlobalTracker(self.association_cfg())
            packet = self.packet(0, [self.obs(1, 0)])
            first_gid = tracker.resolve(packet, [packet])[1][0][1]
            row = self.obs(2, 1)
            row.occluded, row.continuity_ok, row.reliable_age = True, True, .03
            packet = self.packet(1, [self.obs(1, 1), row])
            labels = tracker.resolve(packet, [packet])
            self.assertEqual(labels[1][0][1], first_gid)
            self.assertFalse(labels[2])
            self.assertEqual(tracker.next_gid, 2)

        def test_initially_occluded_local_track_never_gets_a_new_id(self):
            manager = GlobalTracker(self.association_cfg())
            gids = set()
            for f in range(12):
                row = self.obs(1, f)
                row.occluded = True
                row.feature = row.world = None
                row.quality = row.position_quality = 0.
                packet = self.packet(f, [row])
                gids.update(gid for _, gid, _ in manager.resolve(packet, [packet])[1])
            self.assertFalse(gids)
            self.assertEqual(manager.next_gid, 1)

        def test_ambiguous_fragments_do_not_get_association_hints(self):
            box = np.array([130., 100., 180., 300.])
            tracks = [SimpleNamespace(track_id=i, reliable_box=box.copy(), reliable_time=0.,
                                      box_velocity=np.zeros(4)) for i in (1, 2)]
            tracker = SimpleNamespace(tracked_stracks=tracks, lost_stracks=[])
            rows = [self.obs(1, 1, lid=i) for i in (1, 2)]
            for row in rows:
                row.bbox = np.array([130., 100., 180., 195.])
            prepare_occlusions(tracker, rows, self.association_cfg())
            for row in rows:
                self.assertTrue(row.occluded)
                self.assertTrue(row.partial)
                self.assertEqual(row.hint_local_id, -1)
                self.assertIsNone(row.association_bbox)
                self.assertIsNone(row.feature)
                self.assertIsNone(row.world)

        def test_real_partial_boxes_keep_local_and_global_identity(self):
            if not include_botsort:
                self.skipTest("--test-botsort requires torch/ultralytics")
            cfg = self.association_cfg()
            locals_ = {c: make_local_tracker(30, cfg) for c in (1, 2)}
            guard, manager, buffer = SegmentGuard(cfg), GlobalTracker(cfg), FixedLagBuffer(.3, 30)
            image = np.zeros((720, 1280, 3), np.uint8)
            local_ids, global_ids, output = {1: set(), 2: set()}, set(), []
            for f in range(50):
                packet = self.packet(f, [])
                for c in (1, 2):
                    row = self.obs(c, f)
                    if c == 1 and 10 <= f < 30:
                        if f % 2:
                            row.bbox[1] = 200.  # lower body only
                        else:
                            row.bbox[3] = 200.  # upper body only
                        row.feature = np.eye(4, dtype=np.float32)[1]
                        row.world = np.array([99., 99.])
                        if 15 <= f < 20:
                            row.confidence = .15  # exercise ByteTrack's second stage
                    raw_box = row.bbox.copy()
                    prepare_occlusions(locals_[c], [row], cfg)
                    rows = local_update(locals_[c], [row], image)
                    self.assertEqual(len(rows), 1)
                    guard.update(rows, f / 30)
                    packet.observations[c] = rows
                    local_ids[c].add(rows[0].local_id)
                    np.testing.assert_array_equal(row.bbox, raw_box)
                    if c == 1 and 10 <= f < 30:
                        self.assertTrue(row.partial)
                        self.assertTrue(row.continuity_ok)
                        self.assertIsNone(row.feature)
                        self.assertIsNone(row.world)
                        self.assertEqual(row.epoch, 0)
                        np.testing.assert_allclose(locals_[c].tracked_stracks[0].smooth_feat, np.eye(4)[0])
                        self.assertGreater(row.tracking_bbox[3] - row.tracking_bbox[1], 190.)
                buffer.push(packet)
                while buffer.ready():
                    output.append(buffer.pop(manager))
            while buffer.packets:
                output.append(buffer.pop(manager, flush=True))
            for _, labels, _, _ in output:
                for rows in labels.values():
                    global_ids.update(gid for _, gid, _ in rows)
            self.assertEqual([len(v) for v in local_ids.values()], [1, 1])
            self.assertEqual(len(global_ids), 1)

        def test_real_missing_has_no_output_and_does_not_refresh_identity(self):
            if not include_botsort:
                self.skipTest("--test-botsort requires torch/ultralytics")
            cfg = self.association_cfg(max_gap=.8, local_buffer_seconds=.8, identity_memory_seconds=300.)
            local, guard, manager = make_local_tracker(30, cfg), SegmentGuard(cfg), GlobalTracker(cfg)
            image = np.zeros((720, 1280, 3), np.uint8)
            for f in range(40):
                rows = [self.obs(1, f)] if f < 4 else []
                prepare_occlusions(local, rows, cfg)
                rows = local_update(local, rows, image)
                guard.update(rows, f / 30)
                packet = self.packet(f, rows)
                labels = manager.resolve(packet, [packet])
                if f >= 4:
                    self.assertFalse(labels[1])
                    for identity in manager.identities.values():
                        self.assertAlmostEqual(identity.last, 3 / 30)
            self.assertEqual(manager.next_gid, 2)
            self.assertTrue(manager.identities)  # Appearance survives the local-motion buffer.
            packet = self.packet(9004, [])
            self.assertEqual(manager.resolve(packet, [packet]), {1: [], 2: []})
            self.assertFalse(manager.identities)  # But the appearance cache is bounded too.

        def test_real_reappearance_after_missing_keeps_identity(self):
            if not include_botsort:
                self.skipTest("--test-botsort requires torch/ultralytics")
            cfg, guard = self.association_cfg(), SegmentGuard(self.association_cfg())
            local, manager = make_local_tracker(30, cfg), GlobalTracker(cfg)
            image = np.zeros((720, 1280, 3), np.uint8)
            ids = set()
            for f in range(25):
                rows = [] if 6 <= f < 12 else [self.obs(1, f)]
                prepare_occlusions(local, rows, cfg)
                rows = local_update(local, rows, image)
                guard.update(rows, f / 30)
                packet = self.packet(f, rows)
                labels = manager.resolve(packet, [packet])
                if 6 <= f < 12:
                    self.assertFalse(labels[1])
                    continue
                self.assertEqual(len(labels[1]), 1)
                obs, gid, _ = labels[1][0]
                ids.add((obs.local_id, gid))
            self.assertEqual(len(ids), 1)

        def test_real_two_people_overlap_stop_and_separate(self):
            if not include_botsort:
                self.skipTest("--test-botsort requires torch/ultralytics")
            cfg, guard = self.association_cfg(), SegmentGuard(self.association_cfg())
            local, manager = make_local_tracker(30, cfg), GlobalTracker(cfg)
            image = np.zeros((720, 1280, 3), np.uint8)
            ids, count = {0: set(), 1: set()}, 0
            for f in range(60):
                shift = min(35, max(0, (f - 5) * 2), max(0, (55 - f) * 2))
                rows = []
                for person in (0, 1):
                    row = self.obs(1, f, person=person)
                    x = 100 + 90 * person + shift * (1 if person == 0 else -1)
                    row.bbox = np.array([x, 100, x + 50, 300.])
                    rows.append(row)
                owners = {id(row): p for p, row in enumerate(rows)}
                prepare_occlusions(local, rows, cfg)
                count += sum(row.occluded for row in rows)
                output = local_update(local, rows, image)
                guard.update(output, f / 30)
                packet = self.packet(f, output)
                labels = manager.resolve(packet, [packet])
                self.assertEqual(len(labels[1]), 2)
                for row, gid, _ in labels[1]:
                    ids[owners[id(row)]].add((row.local_id, gid))
            self.assertGreater(count, 0)
            self.assertEqual([len(v) for v in ids.values()], [1, 1])
            self.assertNotEqual(ids[0], ids[1])

        def test_render_draws_only_one_raw_box_and_id(self):
            import cv2
            from unittest.mock import patch
            row = self.obs(1, 0)
            original = row.bbox.copy()
            row.tracking_bbox = row.bbox + np.array([-5, -5, 5, 5])
            row.occluded = True
            image = np.zeros((400, 300, 3), np.uint8)
            with patch.object(cv2, "rectangle", wraps=cv2.rectangle) as rectangles, \
                 patch.object(cv2, "putText", wraps=cv2.putText) as texts:
                output = render(image, [(row, 1, "tentative")], 1, 0, 0.)
            outlines = [call.args for call in rectangles.call_args_list if call.args[4] > 0]
            self.assertEqual(len(outlines), 1)
            self.assertEqual(outlines[0][1:3], (tuple(original[:2].astype(int)), tuple(original[2:].astype(int))))
            self.assertEqual(texts.call_args_list[0].args[1], "ID 1")
            self.assertGreater(np.count_nonzero(output), 0)
            self.assertEqual(np.count_nonzero(image), 0)
            np.testing.assert_array_equal(row.bbox, original)

        def test_render_never_draws_a_box_without_detection(self):
            row = self.obs(1, 0)
            row.tracking_bbox, row.bbox = row.bbox.copy(), None
            image = np.zeros((400, 300, 3), np.uint8)
            actual = render(image, [(row, 1, "single_view")], 1, 0, 0.)
            np.testing.assert_array_equal(actual, render(image, [], 1, 0, 0.))

        def test_duplicate_filter_removes_only_near_identical_boxes(self):
            tracker = SimpleNamespace(tracked_stracks=[], lost_stracks=[], duplicate_detections_removed=0)
            boxes = np.array([[100, 100, 150, 300], [100.2, 100, 150.2, 300],
                              [100, 100, 150, 200], [130, 100, 180, 300]], np.float32)
            keep = deduplicate_detections(boxes, [.7, .9, .8, .8], tracker, 0., self.association_cfg())
            self.assertEqual(keep.tolist(), [1, 2, 3])
            self.assertEqual(tracker.duplicate_detections_removed, 1)

        def test_duplicate_filter_preserves_two_confirmed_people(self):
            tracks = [SimpleNamespace(track_id=i, is_activated=True, last_observation_time=0.,
                                      tlwh=np.array([100., 100., 50., 200.])) for i in (1, 2)]
            tracker = SimpleNamespace(tracked_stracks=tracks, lost_stracks=[], duplicate_detections_removed=0)
            boxes = np.array([[100, 100, 150, 300], [100.2, 100, 150.2, 300]], np.float32)
            keep = deduplicate_detections(boxes, [.9, .7], tracker, 1 / 30, self.association_cfg())
            self.assertEqual(keep.tolist(), [0, 1])
            self.assertEqual(tracker.duplicate_detections_removed, 0)

        def test_real_duplicate_detections_do_not_spawn_second_identity(self):
            if not include_botsort:
                self.skipTest("--test-botsort requires torch/ultralytics")
            cfg, guard = self.association_cfg(), SegmentGuard(self.association_cfg())
            local, manager = make_local_tracker(30, cfg), GlobalTracker(cfg)
            image = np.zeros((720, 1280, 3), np.uint8)
            ids = set()
            for f in range(15):
                rows = [self.obs(1, f), self.obs(1, f)]
                rows[1].bbox += np.array([.2, 0, .2, 0])
                rows[1].confidence = .6
                keep = deduplicate_detections(np.array([r.bbox for r in rows]),
                                             [r.confidence for r in rows], local, f / 30, cfg)
                rows = [rows[i] for i in keep]
                prepare_occlusions(local, rows, cfg)
                self.assertFalse(rows[0].occluded)
                output = local_update(local, rows, image)
                guard.update(output, f / 30)
                packet = self.packet(f, output)
                labels = manager.resolve(packet, [packet])
                self.assertEqual(len(labels[1]), 1)
                row, gid, _ = labels[1][0]
                ids.add((row.local_id, gid))
            self.assertEqual(len(ids), 1)
            self.assertEqual(local.duplicate_detections_removed, 15)

    suite = unittest.defaultTestLoader.loadTestsFromTestCase(Tests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())
