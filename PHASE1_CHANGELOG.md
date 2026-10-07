# 1차 수정안 변경 및 검토 기록

파란색 문구는 이번 복사본에서 새로 반영하거나 확인한 사항입니다.

## 결과

<span style="color:#0070C0">작업 폴더는 Drive의 `integrated_fall_inference_module` 실제 코드가 들어 있는 항목에서 복제했습니다. 원본은 읽기 전용으로 유지했고, 최종 ZIP도 Drive에 올리지 않았습니다.</span>

<span style="color:#0070C0">TCN 추론 16F/8F, STDS 64F/16F, STDS 창당 DBN 1회 갱신을 config에서 검증합니다. 학습용 TCN stride4 설정은 별도 설정이며 변경하지 않았습니다.</span>

<span style="color:#0070C0">`auto/cpu/cuda/cuda:index` 장치 해석을 공통화했습니다. `auto`는 가용 GPU를 선택하고, 명시된 CUDA 장치가 없거나 인덱스가 범위를 벗어나면 CPU로 조용히 전환하지 않고 오류를 냅니다. YOLO와 PyTorch가 같은 장치를 사용합니다.</span>

<span style="color:#0070C0">영상 0프레임·잘못된 FPS/해상도·잘못된 2D/3D shape와 비유한 값을 검사합니다. 비디오 capture 해제를 finally로 보장합니다. NPZ는 알려진 skeleton 키만 허용하고, 빈 NPZ·멀티비디오 경계 메타데이터를 구별해 거부하며 `(0,17,3)`은 유효한 빈 skeleton 입력으로 유지합니다.</span>

<span style="color:#0070C0">결과 JSON에 schema_version, UTC 생성시각, status, pose_quality를 더했습니다. STDS raw 확률은 결과에 그대로 기록하며 clipping은 DBN 입력 계산에만 씁니다. 결과 쓰기는 `allow_nan=False`, atomic replace, 기본 덮어쓰기 거부 방식입니다.</span>

<span style="color:#0070C0">추론 중 DBN 상태 reset과 순차 갱신은 엔진 lock으로 보호합니다. 매 입력 시퀀스 시작 시 DBN 상태를 초기화합니다. 검출률 0.5 미만은 잠정 기준으로 `insufficient_pose` 상태만 부여하며 분류 라벨은 바꾸지 않습니다. 검출률 기준은 실제 검증으로 보정할 필요가 있습니다.</span>

<span style="color:#0070C0">세 전처리 annotation builder에서 빈 프레임 값은 None으로 처리하고, 부분 구간·비정수·음수·역전 구간은 실패시킵니다. DBN train/eval CLI의 인자 불일치를 positional `project_root repeat_dir split_json`으로 고쳤고, unified 데이터의 프로젝트 루트 경로도 인식합니다. STDS patience를 저장 설정의 15로 맞췄습니다.</span>

## 원본과 불일치/주의 사항

<span style="color:#0070C0">Drive 원본의 `MANIFEST.json`은 58개 파일만 목록화했지만 실제 파일은 81개였습니다. 원본 MANIFEST는 `PHASE1_ORIGINAL_MANIFEST.json`으로 보존했고, 전체 81개 파일 크기/SHA-256은 `PHASE1_ORIGINAL_INVENTORY.json`에 기록했습니다.</span>

<span style="color:#0070C0">기존 MANIFEST가 기대한 `Colab_Run_Module.ipynb`는 1,746 bytes였으나 실제 원본은 7,278 bytes였습니다(실제 SHA-256 `7c80ce279788a5803df23cc14bf5d374374146eec57d2160c07b448e787ee174`). 원본 MANIFEST는 이 수정안에서 덮어쓰지 않았습니다.</span>

<span style="color:#0070C0">원본에서 내려받은 다섯 모델 파일은 최종 수정본에서도 동일 크기와 SHA-256입니다. 결과 검증은 `PHASE1_FINAL_MANIFEST.json`의 `protected_models`에서 확인할 수 있습니다.</span>

## 실행 검증과 제한

<span style="color:#0070C0">통과: Python 전체 compileall 및 계약 테스트(창 길이 0/15/16/63/64/65/80/448, NPZ empty-vs-zero-length, malformed value, multi-video metadata).</span>

<span style="color:#0070C0">미실행: PyTorch 모델 checkpoint load/forward, YOLO 실제 영상 추론, 원본 parity/정확도 비교, CUDA 선택 경로. 현재 Python 환경에 PyTorch가 없습니다. 따라서 실제 영상 정확도와 모델 구조/가중치의 동작 호환성은 검증 완료로 표시하지 않습니다.</span>

<span style="color:#0070C0">남은 검증: TCN stride4 clip 원본과 시작 프레임 audit(복사본에 해당 원본 clip 묶음이 없음), GPU checkpoint forward/실영상 성능 평가(PyTorch 없음), confidence 기준의 실데이터 보정. `check_module.py`는 파일/모델 해시, 모델 로드/합성 forward, 선택적 영상 E2E를 분리합니다. 영상 정확도는 라벨된 ground truth가 필요하므로 계산하지 않습니다.</span>


<span style="color:#0070C0">`Video (14).mp4`는 OpenCV로 448/448 frame decode, 30 FPS, 1280×720임을 확인했습니다. 이것은 입력 디코딩 확인일 뿐 모델 E2E 추론/정확도 결과가 아닙니다. STDS 내부 TCN과 독립 TCN의 state dict 동기화를 런타임 초기화에서 검사하도록 추가했습니다(현재 환경에서 PyTorch 검증은 미실행).</span>
