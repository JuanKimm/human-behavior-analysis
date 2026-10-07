# Google Colab 실행 가이드

## 원칙
- CPU와 GPU runtime을 각각 검증할 수 있습니다.
- seed42 test 영상은 Gate-3 전에는 사용하지 않습니다.
- Raw-video 재현성을 위해 preprocessing trio를 exact pin합니다.

## 1. 프로젝트 준비
Drive mount 후 candidate를 풀고 프로젝트 루트로 이동합니다.

## 2. 환경 정렬
Colab에 기존 OpenCV/Ultralytics 변형이 있으면 먼저 제거한 뒤 프로젝트를 설치합니다.
```bash
pip uninstall -y ultralytics opencv-python opencv-python-headless
pip install -e .
```
검증값:
- NumPy `2.2.6`
- OpenCV `4.11.0` (`opencv-python==4.11.0.86`)
- Ultralytics `8.4.70`

Torch는 Colab runtime이 제공하는 CPU/CUDA build를 사용합니다. 현재 검증된 CUDA 예시는 `torch 2.11.0+cu128`, `torchvision 0.26.0+cu128`입니다.

## 3. 사전검사
```bash
python check_module.py --device cpu
# 또는 GPU runtime
python check_module.py --device cuda
```
환경 계약 mismatch이면 공식 검증을 중단합니다.

## 4. Validation raw-video 실행
`run_video.py`를 Validation 영상과 고유 `ENV::video_name`으로 실행합니다. Test 43은 Gate-3 전 금지입니다.

## 5. 증거 보존
생성된 run 폴더의 `_evidence`와 전체 output bundle을 보존합니다.

환경 통제 근거는 `docs/ENVIRONMENT_REPRODUCIBILITY.md`를 참고합니다.
