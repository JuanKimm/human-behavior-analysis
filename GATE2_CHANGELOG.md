# Gate-2 approved remediation - candidate_001

This candidate applies the approved `DD-R01_0007 ... v0.1` Batch A+B+C scope without modifying the original V2 archive.

## Core changes
- Output run bundle with monotonic `_run_001`, `_run_002` directories and no default overwrite.
- `predict` / `evaluate` separation; GT labels are loaded only after inference. seed42 test is locked until Gate-3.
- `frame_timeline.csv`, `transition_summary.csv`, `metrics_summary.json`, `result.json`, and `_evidence/` artifacts.
- ST-DS pre-CRF and post-CRF research outputs; DBN remains driven by pre-CRF observations.
- `research` and `product` presentation profiles.
- Public `dd_r01` API and packaging metadata.
- Explicit `insufficient_length` / `partial` model status.
- Detached release manifest and fail-fast integrity checker.
- Unique `env::video_name` evaluation identity and hash-suffixed arbitrary predict identity.
- Test-set protection, Local/Colab guides, parity unique-key fix, expanded contract tests.
- TCN stride4 reconstruction/provenance tool; exact split clip counts reproduced from unified data.
- Release cleanup removes legacy test fixture/output and Python cache files.

## Deferred to Phase 3 execution evidence
- Raw-video YOLO -> VideoPose3D -> models -> overlay E2E on Local CPU, Colab CPU, Colab GPU.
- Full Validation split execution and performance/memory evidence.
- Final seed42 test 43 execution only after Gate-3 approval.


## S3 environment-reproducibility follow-up - candidate_002
- Exact raw-video preprocessing contract added: NumPy 2.2.6 / OpenCV 4.11.0.86 / Ultralytics 8.4.70.
- Torch/torchvision remain platform-specific within validated ranges because aligned Windows CPU and Colab CUDA produced semantic parity.
- Raw-video entry path now fails fast when the preprocessing contract is not satisfied.
- Provenance now records Ultralytics, torchvision and scikit-learn versions.
- Local/Colab guides and environment reproducibility evidence were updated.
- No model/checkpoint replacement is authorized by this change.
