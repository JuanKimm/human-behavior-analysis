# 오프라인 MTMCT + 낙상 분석 통합

기준: `develop` 커밋 `e8b682dda852b771db2bd4cd62c99e036b004d2a`.

## 이번 단계의 범위

새 진입점은 `run_mtmct_fall.py`입니다. 이미 동기화된 두 영상에서 YOLO26x-Pose를 한 번씩 실행하고, 같은 검출 결과의 bbox를 기존 MTMCT에 전달합니다. 확정된 global ID와 원본 관절을 연결한 뒤, 사람·카메라별 시퀀스를 VideoPose3D와 기존 TCN/ST-DS/DBN에 전달합니다.

기존 `mtmct_delayed_realtime_v4-2-2.py`, `run_video.py`, `run_3d_npz.py`의 실행 경로는 유지합니다. 기존 MTMCT의 변경은 `Observation.detector_index` 추가와 검출 필터링 전 행 번호 보존뿐입니다. 추적·ReID·매칭·partial 판정·고정 지연 버퍼·보정 로직은 수정하지 않습니다. 기존 MTMCT CLI는 bbox 전용 모델을 요구하므로, pose 통합은 새 진입점으로 실행해야 합니다.

알고리즘을 유지해도 bbox 검출기를 YOLO26x-Pose로 바꾸면 추적 정확도와 속도는 달라질 수 있습니다. 기존 결과와 정확도가 동일하다는 의미는 아닙니다.

## 모듈과 입출력

| 모듈 | 역할 |
| --- | --- |
| `src/mtmc_fall/contracts.py` | 검출 관절 스냅샷, 사람·카메라별 연속 시퀀스 계약 |
| `video.py` | 두 입력을 같은 프레임 번호로 순차 디코딩 |
| `tracking.py` | 기존 MTMCT 호출, 필터링 후에도 원본 관절 행 연결 |
| `fall.py` | 공유 모델로 시퀀스별 추론, 카메라 결과 선택 |
| `output.py` | 2D/3D 관절 및 결과 저장, 최종 영상 생성 |
| `environment.py` | 직접 의존성 버전과 torchreid 커밋 검증 |
| `pipeline.py` | 모델 초기화와 오프라인 처리 순서 조정 |

`TrackedPose`의 schema는 `mtmc-fall/1`입니다.

| 필드 | 의미 |
| --- | --- |
| `frame_index`, `timestamp` | 0부터 시작하는 원본 프레임, `frame_index / fps` 초 |
| `camera_id`, `global_id` | 카메라 1/2, 확정된 사람 ID |
| `local_id`, `epoch` | 기존 MTMCT의 로컬 트랙·구간 |
| `detector_index` | 필터링 전 YOLO Results의 bbox/keypoints 행 번호 |
| `image_size` | 원본 `(width, height)` |
| `bbox`, `detection_confidence` | 원본 bbox 픽셀 좌표 및 검출 신뢰도 |
| `keypoints`, `confidence` | COCO17 원본 픽셀 `(17,2)`, 관절 신뢰도 `(17,)` |
| `partial`, `occluded` | 추적기 판정 기록; 통합부의 낙상 제외 조건으로 사용하지 않음 |

2D 좌표는 YOLO의 원본 영상 좌표입니다. bbox crop 좌표나 MTMCT 바닥 좌표를 넣지 않습니다. VideoPose3D 내부에서 원본 가로·세로로 정규화하며, 출력 `(T,17,3)`은 기존 학습 계약대로 추가 root-centering·축 변환·스케일 정규화 없이 전달합니다. MTMCT의 바닥 평면 좌표는 사람 매칭 용도로만 사용합니다.

global ID는 관절 수집 단계부터 포함됩니다. 낙상 결과에 새 ID를 부여하는 것이 아니라, 해당 시퀀스의 ID를 그대로 보존합니다. 이미 출력한 ID를 소급 변경하지 않습니다.

## 사람별 상태와 카메라 선택

- 모델은 실행당 한 번 로드합니다. 동일 가중치를 사용하는 시퀀스 추론은 순차 실행합니다.
- `(global ID, camera)`별 시퀀스를 만듭니다. 다른 사람이나 카메라의 2D/3D 좌표를 한 시퀀스에 섞지 않습니다.
- 기존 낙상 엔진은 각 `infer_skeleton_sequence()` 호출 시작 시 DBN 상태를 초기화합니다. 같은 ID라도 다른 카메라의 DBN 이력은 섞이지 않습니다.
- 관측 누락, 로컬 ID/epoch 변경, 영상 크기 변경 시 연속 구간을 나눕니다. 보간·이전 관절 재사용은 하지 않습니다. 이는 구간을 잘못 이어 붙이지 않기 위한 최소 경계 처리입니다. 가림 복구, 불확실 구간의 유지·복원 정책은 후속 검토 대상입니다.
- `partial` 자체를 이유로 낙상 입력을 제거하지 않습니다. 다만 기존 추적기가 관측을 제거하거나 ID를 확정하지 못하면 ID가 붙은 관절도 생성되지 않아 구간이 끊길 수 있습니다. 이 간접 영향은 기존 추적 동작을 유지한 결과입니다.
- 각 카메라의 DBN 판정을 모두 보관합니다. 같은 source frame/global ID에서 `status=ok`인 결과 중 **판정에 사용한 창의 유효 관절 비율 → 평균 관절 신뢰도 → 카메라 번호가 작은 순서**로 표시 결과를 선택합니다.
- bbox 신뢰도나 Danger 확률이 높다는 이유로 카메라를 선택하지 않습니다. 적격 결과가 없으면 `unavailable`/`N/A`이며 정상으로 간주하지 않습니다.
- 카메라마다 추적 시작점이 달라 판정 창의 시작·끝이 다를 수 있습니다. 각 카메라의 `window_start/end`, `decision_frame`을 기록합니다. 현재 선택법은 초기 정책이며 정확도 검증을 통한 확정은 남아 있습니다.

공개 `infer_skeleton3d()` API는 변경하지 않습니다. 새 통합부는 관절 신뢰도를 전달하기 위해 기존 내부 엔진의 `infer_skeleton_sequence()`를 호출합니다.

## 시간과 입력 조건

두 영상은 **동일 FPS·동일 프레임 수·이미 동기화된 영상**이어야 합니다. 새 reader는 각 카메라에서 매 프레임 한 번씩 읽습니다. 프레임 건너뛰기·복제·offset·drift 보정·짧은 영상에 맞춘 잘라내기를 수행하지 않습니다. 메타데이터 불일치나 중간 디코딩 실패 시 오류로 종료합니다. 동일 FPS와 길이만으로 실제 장면 동기화를 증명할 수는 없으므로 입력을 사전에 맞춰야 합니다.

기존 `color7` 파일명은 MTMCT 원본 보정표에 등록돼 있습니다. 새 스크립트의 기본 경로는 자리표시자이며, **보정 전 원본을 그대로 사용하라는 뜻이 아닙니다.** 실제 동기화된 영상 경로로 교체하세요.

기존 캘리브레이션은 두 원본 카메라의 화각·배치와 1280×720을 전제로 합니다. 다른 카메라에는 캘리브레이션을 별도로 정해야 합니다. 영상 크기만 같다고 동일 캘리브레이션이 성립하지 않습니다.

TCN 16프레임/stride 8, ST-DS·DBN 64프레임/stride 16을 유지합니다. ST-DS의 pre-CRF 확률을 DBN에 전달하는 원본 동작도 유지합니다. 첫 DBN 창 이전은 warmup/N/A입니다. 이후에는 원본 `hold_last` 규칙으로 다음 판정까지 유지하며, 구간을 넘어 유지하지 않습니다. 입력 FPS는 리샘플링하지 않습니다. 학습 영상의 시간 간격과 실제 입력 FPS의 적합성은 별도 확인 대상입니다.

VideoPose3D는 비인과적 243프레임 receptive field 모델로 미래 문맥을 사용합니다. 이번 구현은 추적 완료 후 전체 시퀀스를 변환하는 오프라인 방식입니다. MTMCT의 0.3초 지연이 전체 낙상 판단 지연을 의미하지 않습니다. 길이에 비례해 관절/결과 메모리가 증가하고, 3D 모델도 전체 시퀀스를 한 번에 처리하므로 매우 긴 영상은 메모리 한계가 있을 수 있습니다.

## 환경 준비

Python 3.12와 별도 가상환경을 권장합니다. Windows 예시:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install -r requirements-integration.txt
.venv\Scripts\python -m pip install --no-deps --no-build-isolation "torchreid @ git+https://github.com/KaiyangZhou/deep-person-reid.git@f8cd150fdf77e8d9e1ed143b7f308c2c609ded50"
```

torchreid 설치에는 Git과 C 컴파일러가 필요합니다. Windows는 MSVC Build Tools, Linux는 GCC/G++ 등을 준비하세요. Linux에서 컴파일러가 잘못 지정돼 있으면 마지막 명령 앞에 `CC=gcc CXX=g++`를 붙일 수 있습니다. `--no-deps`는 불필요한 학습 의존성 설치를 피하기 위한 옵션이며 필요한 실행 의존성은 integration requirements에 포함했습니다.

핵심 직접 의존성은 numpy 2.2.6, opencv-python 4.11.0.86, ultralytics 8.4.70, torch 2.13.0, torchvision 0.28.0, scikit-learn 1.8.0, lap 0.5.12로 고정하고 실행 시 확인합니다. torchreid는 위 Git 커밋을 검증합니다. CPU/CUDA wheel, 드라이버, 전체 전이 의존성까지 완전히 고정한 lock 파일은 아닙니다. torch/torchvision의 플랫폼 접미사(`+cpu`, `+cu130` 등)는 허용합니다. GPU 환경은 별도 검증이 필요합니다.

가중치는 Git에 포함하지 않습니다. 기존 압축파일의 다음 파일을 동일 경로에 복사하세요.

- `assets/yolo/yolo26x-pose.pt`
- `assets/videopose3d/pretrained_h36m_detectron_coco.bin`
- `checkpoints/tcn/best.pt`
- `checkpoints/stds/best.pt`
- `checkpoints/dbn/final_dbn.npz`

OSNet은 기존 MTMCT가 사용하는 `weights/osnet_x1_0_msmt17.pth`를 준비합니다. 공식 파일 URL은 기존 MTMCT의 `REID_URL`에 있으며, 기대 SHA256은 `48df972f72887b95cf3b43b3a07c3a7d2398381aea0f9cae64a7ef11d512b727`입니다. 통합 실행 중 가중치를 자동 다운로드하지 않습니다.

## 실행과 결과

`run_mtmct_fall.py` 상단에서 두 영상 경로, 가중치, 출력 폴더, `DEVICE`를 설정합니다.

```bash
python run_mtmct_fall.py
```

`outputs/mtmc_fall/run_.../`에 다음을 저장합니다.

| 결과 | 내용 |
| --- | --- |
| `tracked_poses.jsonl` | 빈 프레임을 포함한 원본 프레임별 관절·ID·추적 진단 |
| `events.jsonl`, `calibration.json` | 기존 매칭 이벤트, 캘리브레이션 |
| `sequences/g<ID>_c<CAM>_f<START>.npz` | 원본 2D, 신뢰도, 3D, source frame/timestamp |
| 같은 이름의 `.json` | 시퀀스별 원본 TCN/ST-DS/DBN 출력 |
| `risks.jsonl` | 카메라별 결과와 선택된 global ID별 결과 |
| `combined_fall.mp4` | 양쪽 영상의 global ID와 선택된 위험 상태·카메라 |
| `summary.json` | 성공/실패, 프레임·ID 수, 실행환경·가중치/소스 해시 |

시퀀스 JSON의 모델 창 인덱스는 **구간 내 상대 인덱스, 양끝 포함**입니다. `source_start_frame`을 더하면 원본 프레임 번호가 됩니다. `risks.jsonl`의 창과 판정 번호는 원본 절대 프레임 번호입니다. 최종 영상은 원본 영상을 두 번째로 읽어 그리며, 표시된 영상을 다시 모델 입력으로 사용하지 않습니다.

## 검증 방법과 한계

```bash
python mtmct_delayed_realtime_v4-2-2.py --self-test --test-botsort
python -m unittest discover -s tests -p "test_*.py"
python tools/smoke_mtmct_integration.py
python tools/smoke_mtmct_integration.py --real-yolo
```

기본 smoke는 84프레임의 합성 영상과 명시적 pose fixture를 사용하고, OSNet·BoT-SORT·global matching·VideoPose3D·낙상 모델은 실제 구현을 실행합니다. 두 사람, 두 카메라의 네 시퀀스와 결과 영상 프레임 수를 확인합니다. `--real-yolo`는 실제 YOLO26x-Pose로 2프레임 빈 영상을 처리해 무검출 경로를 확인합니다. 두 검사를 실제 사람 영상의 정확도 검증으로 해석해서는 안 됩니다.

이번 통합에서는 모델 재학습·임계값 최적화·보호된 최종 test split의 성능 평가는 수행하지 않습니다. 실제 다인 영상의 ID switch, 낙상 민감도/오탐, 카메라 선택 효과, 처리속도는 별도 측정해야 합니다. 원본 candidate 검증 기록은 그대로 두고, 통합 검증 기록은 `environment/integration_validation.json`에 구분합니다.
