#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
VideoPose3D causal 추론 실행 스크립트
입력: Keypoints_2D/video(N).npz (COCO 17 keypoints, 2D)
      ← pretrained_h36m_detectron_coco.bin은 COCO 17 형식을 입력으로 받음
출력: Keypoints_3D/video(N).npz (H36M 17 keypoints, 3D)

먼저 setup_videopose3d.py를 실행해서
model.py와 checkpoint를 준비해야 합니다.

사용 모델: TemporalModel (공식 facebookresearch/VideoPose3D)
체크포인트: pretrained_h36m_detectron_coco.bin
  - COCO 17 관절 (x,y) 입력 → H36M 17 관절 (x,y,z) 출력
  - architecture: -arc 3,3,3,3,3 (243 frames, non-causal)
"""

import torch
import numpy as np
import os
import glob
import sys

# model.py가 같은 폴더에 있어야 함 (setup_videopose3d.py로 다운로드)
try:
    from model import TemporalModel
except ImportError:
    print("❌ model.py를 찾을 수 없습니다.")
    print("먼저 setup_videopose3d.py를 실행하세요.")
    sys.exit(1)

# =============================================
# 설정
# =============================================
BASE_PATH      = r"C:\Users\User\OneDrive\data\RoseLab+Le2i+3AI"
CHECKPOINT_PATH = os.path.join("checkpoint_3ai", "pretrained_h36m_detectron_coco.bin")

FOLDERS = [
    #"Coffee_room_01",
    #"Coffee_room_02",
    #"Home_01",
    #"Home_02",
    #"Lecture_room",
    #"Office",
    #"RoseLab_S001"
    "3AI_Office_01"
]

# 공식 체크포인트 기본 아키텍처 (243 프레임, non-causal)
FILTER_WIDTHS = [3, 3, 3, 3, 3]
CAUSAL        = False   # 공식 체크포인트는 non-causal로 학습됨
CHANNELS      = 1024
NUM_JOINTS    = 17
IN_FEATURES   = 2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# H36M 좌우 대칭 관절 (테스트 시 좌우 반전 augmentation에 사용,
# 첨부 이미지 기준 인덱스)
JOINTS_LEFT  = [1, 2, 3, 14, 15, 16]   # L.Hip,L.Knee,L.Foot,L.Sho,L.Elb,L.Hand
JOINTS_RIGHT = [4, 5, 6, 11, 12, 13]   # R.Hip,R.Knee,R.Foot,R.Sho,R.Elb,R.Hand


def load_model():
    """모델 로드 및 체크포인트 가중치 적용"""
    print(f"모델 생성 중... (device: {DEVICE})")

    model = TemporalModel(
        num_joints_in  = NUM_JOINTS,
        in_features    = IN_FEATURES,
        num_joints_out = NUM_JOINTS,
        filter_widths  = FILTER_WIDTHS,
        causal         = CAUSAL,
        channels       = CHANNELS
    )

    if not os.path.exists(CHECKPOINT_PATH):
        print(f"❌ 체크포인트 없음: {CHECKPOINT_PATH}")
        print(f"setup_videopose3d.py의 안내에 따라 다운로드하세요.")
        sys.exit(1)

    print(f"체크포인트 로드 중: {CHECKPOINT_PATH}")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=True)
    model.load_state_dict(checkpoint['model_pos'])

    model = model.to(DEVICE)
    model.eval()

    receptive_field = model.receptive_field()
    print(f"✓ 모델 로드 완료")
    print(f"  Receptive field: {receptive_field}프레임")
    print(f"  파라미터 수: {sum(p.numel() for p in model.parameters()):,}")

    return model, receptive_field


def normalize_2d(keypoints, width, height):
    """
    VideoPose3D 표준 정규화
    [0, width] x [0, height] 픽셀좌표 -> [-1, 1] 범위로 정규화
    (공식 구현 common/camera.py의 normalize_screen_coordinates 방식)
    """
    kp = keypoints.copy().astype(np.float32)
    kp[..., 0] = kp[..., 0] / width * 2 - 1
    kp[..., 1] = kp[..., 1] / width * 2 - height / width
    return kp


def pad_sequence(keypoints, pad):
    """
    Receptive field만큼 양 끝을 복제(replicate padding)하여
    출력 프레임 수 = 입력 프레임 수가 되도록 함
    (공식 run.py의 evaluate() 방식과 동일)
    """
    return np.pad(keypoints, ((pad, pad), (0, 0), (0, 0)), mode='edge')


def run_inference(model, receptive_field, keypoints_2d, width, height):
    """
    단일 비디오의 2D 시퀀스 전체를 3D로 변환

    keypoints_2d: (N, 17, 2) - 픽셀 좌표
    Returns: (N, 17, 3) - 3D 좌표 (mm 단위가 아닌 정규화 카메라 공간)
    """
    pad = (receptive_field - 1) // 2

    # 정규화
    kp_norm = normalize_2d(keypoints_2d, width, height)

    # 패딩 (양 끝 프레임 복제)
    kp_padded = pad_sequence(kp_norm, pad)

    # 텐서 변환 (batch=1)
    input_tensor = torch.from_numpy(kp_padded).unsqueeze(0).float().to(DEVICE)

    with torch.no_grad():
        output = model(input_tensor)   # (1, N, 17, 3)

    output_3d = output.squeeze(0).cpu().numpy()

    return output_3d


def process_video(npz_path, output_path, model, receptive_field):
    """
    H36M 2D NPZ -> 3D NPZ 변환
    """
    data = np.load(npz_path)

    keypoints_2d = data['keypoints']     # (N, 17, 2)
    confidence   = data['confidence']    # (N, 17)
    detected     = data['detected']      # (N,)
    fps          = data['fps']
    resolution   = data['resolution']    # [width, height]
    frame_count  = data['frame_count']

    width, height = int(resolution[0]), int(resolution[1])
    N = keypoints_2d.shape[0]

    if N < 1:
        return False

    keypoints_3d = run_inference(model, receptive_field, keypoints_2d, width, height)

    np.savez(
        output_path,
        keypoints_3d = keypoints_3d.astype(np.float32),   # (N, 17, 3)
        confidence   = confidence,
        detected     = detected,
        frame_count  = frame_count,
        fps          = fps,
        resolution   = resolution
    )

    return True


# =============================================
# 메인
# =============================================
if __name__ == "__main__":
    print("="*60)
    print("🚀 VideoPose3D 3D 관절 좌표 추론")
    print("="*60)

    model, receptive_field = load_model()

    total_files   = 0
    success_files = 0
    failed_files  = []

    for folder_idx, folder_name in enumerate(FOLDERS, 1):
        folder_path = os.path.join(BASE_PATH, folder_name)
        h36m_path   = os.path.join(folder_path, "Keypoints_2D")   # COCO 17 원본 (pretrained_h36m_detectron_coco.bin 입력 형식)
        out_path    = os.path.join(folder_path, "Keypoints_3D")

        print(f"\n[{folder_idx}/7] {folder_name}")
        print("-" * 50)

        if not os.path.exists(h36m_path):
            print(f"  ❌ Keypoints_2D_H36M 폴더 없음")
            continue

        os.makedirs(out_path, exist_ok=True)

        npz_files = sorted(glob.glob(os.path.join(h36m_path, "*.npz")))

        if not npz_files:
            print(f"  ❌ NPZ 파일 없음")
            continue

        print(f"  파일 수: {len(npz_files)}개")

        for f_idx, npz_path in enumerate(npz_files, 1):
            video_name  = os.path.splitext(os.path.basename(npz_path))[0]
            output_path = os.path.join(out_path, video_name)

            if os.path.exists(output_path + ".npz"):
                success_files += 1
                total_files += 1
                continue

            try:
                process_video(npz_path, output_path, model, receptive_field)
                success_files += 1
            except Exception as e:
                print(f"  ❌ 실패: {video_name} - {e}")
                failed_files.append(npz_path)

            total_files += 1

            if f_idx % 50 == 0 or f_idx == len(npz_files):
                print(f"  진행: {f_idx}/{len(npz_files)}", end='\r')

        print(f"  ✓ 완료: {len(npz_files)}개")

    print(f"\n{'='*60}")
    print(f"✅ 전체 추론 완료!")
    print(f"{'='*60}")
    print(f"총 파일:  {total_files}개")
    print(f"성공:     {success_files}개")
    print(f"실패:     {len(failed_files)}개")

    if failed_files:
        print(f"\n❌ 실패한 파일:")
        for f in failed_files[:10]:
            print(f"   {f}")
        if len(failed_files) > 10:
            print(f"   ... 외 {len(failed_files)-10}개")

    print(f"\n📁 저장 위치: 각 폴더의 Keypoints_3D/")
    print(f"\n💡 NPZ 파일 읽는 법:")
    print(f"   data = np.load('video (1).npz')")
    print(f"   keypoints_3d = data['keypoints_3d']  # (N, 17, 3)")
