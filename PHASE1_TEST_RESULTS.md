# 1차 수정안 테스트 결과

| 검증 | 결과 | 근거 |
|---|---|---|
| Python compileall | 통과 | 전체 프로젝트 Python 파일 컴파일 |
| window schedule 계약 | 통과 | 길이 0, 15, 16, 63, 64, 65, 80, 448; TCN count 0,0,1,6,7,7,9,55; STDS count 0,0,0,0,1,1,2,25 |
| NPZ empty/zero-length 구분 | 통과 | 알려진 skeleton key가 있는 shape `(0,17,3)` 허용, 알려진 키 없는 NPZ는 오류 |
| NPZ malformed/boundary metadata | 통과 | NaN 값 및 unified multi-video 경계 키 거부 |
| Annotation intervals | 통과 | 세 데이터셋 빌더에서 빈 구간, 부분 구간, 0-based inclusive 및 danger override 확인 |
| DBN checkpoint numeric validation | 통과 | 제공된 `final_dbn.npz` 로드, 배열 shape/확률합/공분산 SPD 검증 |
| Unified dataset/split validation | 통과 | 제공 데이터 `(142,942,17,3)`, 290 videos, 203/44/43 split 및 이름/환경/데이터 SHA 연결 검증 |
| Source video decode | 통과 | `Video (14).mp4`: metadata 448 frames, 30 FPS, 1280×720; OpenCV에서 448 frames 실제 decode |
| PyTorch model load/forward | 미실행 | 실행 환경에 `torch` 패키지가 없음 |
| 실제 영상 E2E/parity 정확도 | 미실행 | 영상 디코드만 확인; PyTorch/Ultralytics 실행 의존성이 충족되지 않아 모델 추론/정확도는 실행하지 않음 |
| CUDA/인덱스 경로 | 미실행 | CUDA/PyTorch 런타임 없음 |

단위 테스트 실행 명령: `python -m unittest discover -s tests -p 'test_phase1.py' -v`
