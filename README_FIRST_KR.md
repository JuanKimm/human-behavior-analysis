# DD-R01 낙상 위험 추론 모듈 - Candidate 001

현재 합격 대상은 **파일 기반 batch E2E**입니다. 실시간 streaming은 후속 확장입니다.

## 기본 경로

`Raw Video -> YOLO-Pose -> VideoPose3D -> 3D Skeleton -> TCN / ST-DS -> DBN -> output bundle`

- TCN: 16F / stride 8 / Non-Danger, Danger
- ST-DS: 64F / stride 16 / Normal, Precursor, Danger
- DBN: ST-DS **pre-CRF** observation 사용
- research profile: TCN + ST-DS pre-CRF + ST-DS post-CRF + DBN
- product profile: TCN + DBN 중심 표시

## 가장 쉬운 실행

```bash
python run_video.py --video "C:\\data\\sample.mp4" --mode predict --profile research --device cpu
```

성공하면 `outputs/<video_key>_run_001/`가 생성됩니다. 같은 영상을 다시 실행하면 `_run_002`, `_run_003`으로 증가하며 overwrite하지 않습니다.

## 평가 실행

GT는 모델 추론이 끝난 뒤 evaluator에서만 결합됩니다.

```bash
python run_video.py --video "C:\\data\\val.mp4" --mode evaluate \
  --gt-npz unified_3d_all_envs.npz --video-key "ENV::Video (X)" --device cpu
```

seed42 test 43개는 기본적으로 잠겨 있습니다. `--allow-test`는 **Gate-3 승인 후 최종평가에서만** 사용합니다.

## 산출물

- `*_overlay.mp4` (raw video 실행)
- `frame_timeline.csv`
- `transition_summary.csv`
- `metrics_summary.json`
- `result.json`
- `_evidence/environment.json`
- `_evidence/run_manifest.json`
- `_evidence/validation_status.json`
- `_evidence/output_hashes.json`

## 통합 API

```python
from dd_r01 import infer_video, infer_skeleton3d
```

세부사항: `docs/INTEGRATION_API_KR.md`

## 실행 가이드

- Local/VS Code: `docs/RUN_GUIDE_LOCAL_KR.md`
- Colab: `docs/RUN_GUIDE_COLAB_KR.md`
- Test 보호: `docs/TEST_SET_PROTECTION.md`
- 현재 검증상태: `CURRENT_VALIDATION_STATUS.md`

## 중요한 제한

- VideoPose3D는 non-causal 243F receptive field이므로 현재 candidate를 online real-time이라고 부르지 않습니다.
- 다중인원/R02 tracked-2D adapter는 후속 통합 범위입니다. 현재 R01 합격 기준은 단일인원 독립 E2E입니다.
- `unified_3d_all_envs.npz` GT는 현재 provisional GT이며, Video (67) 정책을 포함한 전체 Label-QA는 별도 dataset revision으로 관리합니다.

## 환경 재현성
Raw-video 공식 실행은 `numpy==2.2.6`, `opencv-python==4.11.0.86`, `ultralytics==8.4.70`의 검증 프로필을 사용합니다. 근거와 Jetson 예외 범위는 `docs/ENVIRONMENT_REPRODUCIBILITY.md`를 참고하십시오.
