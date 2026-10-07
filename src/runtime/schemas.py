from __future__ import annotations
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone

@dataclass
class TCNResult:
    window_start: int
    window_end: int
    label: str
    probabilities: list[float]
    detected_ratio: float | None = None
    status: str = 'ok'
    pose_quality: dict | None = None
    diagnostics: list[str] = field(default_factory=list)

@dataclass
class STDSResult:
    window_start: int
    window_end: int
    pre_crf_label: str
    pre_crf_probabilities: list[float]
    post_crf_label: str
    post_crf_path: list[int] = field(default_factory=list)
    detected_ratio: float | None = None
    status: str = 'ok'
    pose_quality: dict | None = None

@dataclass
class DBNResult:
    window_start: int
    window_end: int
    label: str
    probabilities: list[float]
    t4: float
    sp: float
    sd: float
    stds_probabilities: list[float]
    detected_ratio: float | None = None
    status: str = 'ok'
    pose_quality: dict | None = None

@dataclass
class InferenceResult:
    source: str
    frame_count: int
    fps: float | None
    tcn: list[TCNResult]
    stds: list[STDSResult]
    dbn: list[DBNResult]
    module_version: str | None = None
    device: str | None = None
    environment: dict | None = None
    detected_ratio: float | None = None
    schema_version: str = '2.0'
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    status: str = 'ok'
    model_status: dict = field(default_factory=dict)
    pose_quality: dict | None = None
    diagnostics: list[str] = field(default_factory=list)

    def to_dict(self):
        return asdict(self)
