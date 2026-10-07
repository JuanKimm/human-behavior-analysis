# Local / VS Code 실행 가이드

## 기준 환경
검증 PC: Windows 10 Pro 64-bit, Python 3.12.10, CPU-only PyTorch 2.13.0+cpu. Raw-video 전처리 재현성을 위해 NumPy/OpenCV/Ultralytics는 `requirements.txt`의 exact pin을 사용합니다.

## 1. 프로젝트 설치
프로젝트 루트에서:
```powershell
python -m pip install -e .
```
설치 후 `numpy==2.2.6`, `opencv-python==4.11.0.86`, `ultralytics==8.4.70`인지 확인합니다.

## 2. 사전검사
```powershell
python check_module.py --device cpu
```
환경 계약 또는 manifest/model 검사가 실패하면 다음 단계로 진행하지 않습니다. 공식 검증에서는 `--allow-unvalidated-environment`를 사용하지 않습니다.

## 3. 영상 predict
```powershell
python run_video.py --video "C:\path\video.mp4" --mode predict --profile research --device cpu
```

## 4. Validation evaluate
```powershell
python run_video.py --video "C:\path\validation.mp4" --mode evaluate --profile research --gt-npz unified_3d_all_envs.npz --video-key "ENV::Video (X)" --device cpu
```

## 5. 성공 판정
`_evidence/validation_status.json`의 `status`가 `ok`이고 필수 산출물이 모두 존재해야 합니다. 콘솔의 문자열만으로 성공을 판정하지 않습니다.

환경 재현성 근거는 `docs/ENVIRONMENT_REPRODUCIBILITY.md`를 참고합니다.
