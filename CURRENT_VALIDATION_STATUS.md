# Current validation status

## Candidate status
- Candidate: `candidate_002`
- Purpose: candidate_001의 S3 환경차이 진단 결과를 반영한 PC/Colab preprocessing environment lock
- Protected model/checkpoint contents: unchanged from original V2 / candidate_001
- seed42 test 43: locked until Gate-3

## Evidence inherited from candidate_001
- Local CPU raw-video E2E: PASS
- Colab CPU raw-video E2E: PASS (unaligned preprocessing versions; used for diagnosis)
- Colab GPU raw-video E2E: PASS (unaligned preprocessing versions; used for diagnosis)
- Local CPU vs aligned Colab CUDA environment-control diagnostic: semantic parity PASS after NumPy/OpenCV/Ultralytics alignment

## Candidate_002 delta
Candidate_002 changes only environment reproducibility enforcement, provenance capture, package metadata/guides/tests, and version metadata. Core preprocessing/model algorithms and protected model/checkpoint files are not intentionally changed.

## Required before Gate-3
- candidate_002 manifest/hash integrity regression
- unit/contract tests
- protected model hash parity
- strict environment preflight confirmation on Local CPU and aligned Colab GPU (official run must not use override)

No final seed42 test metrics have been run.
