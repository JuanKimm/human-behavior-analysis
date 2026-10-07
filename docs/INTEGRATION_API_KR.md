# Python 통합 API

설치:
```bash
python -m pip install -e .
```

## raw video inference
```python
from dd_r01 import infer_video
result = infer_video('video.mp4', device='cpu')
```

## 3D skeleton inference
```python
from dd_r01 import infer_skeleton3d
result = infer_skeleton3d(skeleton_3d, device='cpu', fps=30.0)
```

`result`는 CLI core result와 같은 output schema를 사용합니다. 3D skeleton shape는 `(T, 17, 3)`입니다.

고수준 output bundle이 필요하면 `dd_r01.api.run_video_job`, `run_skeleton3d_job`을 사용합니다.

R02 tracked-2D adapter는 현재 필수 API가 아니며 후속 통합 기능입니다.
