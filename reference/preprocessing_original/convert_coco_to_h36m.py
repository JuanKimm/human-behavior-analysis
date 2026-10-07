#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
COCO 17 Keypoints → Human3.6M 17 Keypoints 변환기
YOLO26-pose 결과(NPZ)를 VideoPose3D 입력 형식으로 변환

COCO 17 인덱스:
  0:nose 1:R.eye 2:L.eye 3:R.ear 4:L.ear
  5:R.shoulder 6:L.shoulder 7:R.elbow 8:L.elbow
  9:R.wrist 10:L.wrist 11:R.hip 12:L.hip
  13:R.knee 14:L.knee 15:R.ankle 16:L.ankle

  ※ 주의: 이미지 기준 라벨은 "right/left"이지만
    실제 YOLO/Ultralytics 공식 COCO 순서는
    5:L.shoulder 6:R.shoulder ... 로 좌우가 반대로 표기되는
    경우가 흔함. 본 코드는 Ultralytics 공식 순서를 따름:
    0:nose 1:L.eye 2:R.eye 3:L.ear 4:R.ear
    5:L.shoulder 6:R.shoulder 7:L.elbow 8:R.elbow
    9:L.wrist 10:R.wrist 11:L.hip 12:R.hip
    13:L.knee 14:R.knee 15:L.ankle 16:R.ankle

H36M 17 인덱스 (첨부 이미지 기준):
  0:Bottom torso(Pelvis) 1:L.hip 2:L.knee 3:L.foot
  4:R.hip 5:R.knee 6:R.foot 7:Center torso(Spine)
  8:Upper torso(Thorax) 9:Neck base 10:Center head
  11:R.shoulder 12:R.elbow 13:R.hand(wrist)
  14:L.shoulder 15:L.elbow 16:L.hand(wrist)

직접 매핑 가능: 12개 관절 (어깨/팔/엉덩이/무릎/발목)
계산 필요 관절: 5개
  0(Pelvis)      = (L.hip + R.hip) / 2
  7(Spine)       = (Pelvis + Thorax) / 2
  8(Thorax)      = (L.shoulder + R.shoulder) / 2
  9(Neck base)   = Thorax 그대로 사용 (학계 표준 방식)
                   ※ 많은 논문이 H36M의 "Neck/Nose"를 버리고
                     "Thorax"를 "Neck"으로 매핑함
  10(Center head)= Neck에서 nose 방향으로 "고정 비율"(어깨너비*0.5)
                   만큼 연장. nose 좌표를 그대로 쓰면 고개를
                   들 때 head가 앞으로 쏠려 숙인 것처럼 보이는
                   왜곡이 발생하므로, 방향만 nose를 따르고
                   거리는 인체비율로 고정함
"""

import numpy as np
import os
import glob

# =============================================
# 설정
# =============================================
BASE_PATH = r"C:\Users\User\OneDrive\data\RoseLab+Le2i"

FOLDERS = [
    "Coffee_room_01",
    "Coffee_room_02",
    "Home_01",
    "Home_02",
    "Lecture_room",
    "Office",
    "RoseLab_S001"
]

# COCO 17 (Ultralytics 공식 순서) 인덱스
COCO_NOSE        = 0
COCO_L_SHOULDER  = 5
COCO_R_SHOULDER  = 6
COCO_L_ELBOW     = 7
COCO_R_ELBOW     = 8
COCO_L_WRIST     = 9
COCO_R_WRIST     = 10
COCO_L_HIP       = 11
COCO_R_HIP       = 12
COCO_L_KNEE      = 13
COCO_R_KNEE      = 14
COCO_L_ANKLE     = 15
COCO_R_ANKLE     = 16

# H36M 17 인덱스 (첨부 이미지 기준)
H36M_PELVIS      = 0   # Bottom torso
H36M_L_HIP       = 1
H36M_L_KNEE      = 2
H36M_L_FOOT      = 3
H36M_R_HIP       = 4
H36M_R_KNEE      = 5
H36M_R_FOOT      = 6
H36M_SPINE       = 7   # Center torso
H36M_THORAX      = 8   # Upper torso
H36M_NECK        = 9   # Neck base
H36M_HEAD        = 10  # Center head
H36M_R_SHOULDER  = 11
H36M_R_ELBOW     = 12
H36M_R_HAND      = 13
H36M_L_SHOULDER  = 14
H36M_L_ELBOW     = 15
H36M_L_HAND      = 16

H36M_NAMES = [
    "Pelvis", "L.Hip", "L.Knee", "L.Foot",
    "R.Hip", "R.Knee", "R.Foot", "Spine",
    "Thorax", "Neck", "Head",
    "R.Shoulder", "R.Elbow", "R.Hand",
    "L.Shoulder", "L.Elbow", "L.Hand"
]


def coco17_to_h36m17(coco_kp, coco_conf):
    """
    COCO 17 → H36M 17 변환 (단일 프레임)

    Parameters:
        coco_kp:   (17, 2) - COCO 17 x,y 좌표
        coco_conf: (17,)   - COCO 17 신뢰도

    Returns:
        h36m_kp:   (17, 2) - H36M 17 x,y 좌표
        h36m_conf: (17,)   - H36M 17 신뢰도 (직접 매핑은 원본값,
                              계산값은 관련 관절의 평균 신뢰도)
    """
    h36m_kp   = np.zeros((17, 2), dtype=np.float32)
    h36m_conf = np.zeros((17,),   dtype=np.float32)

    # -----------------------------
    # 1. 직접 매핑 (12개 관절)
    # -----------------------------
    mapping = {
        H36M_L_HIP:      COCO_L_HIP,
        H36M_L_KNEE:     COCO_L_KNEE,
        H36M_L_FOOT:     COCO_L_ANKLE,
        H36M_R_HIP:      COCO_R_HIP,
        H36M_R_KNEE:     COCO_R_KNEE,
        H36M_R_FOOT:     COCO_R_ANKLE,
        H36M_R_SHOULDER: COCO_R_SHOULDER,
        H36M_R_ELBOW:    COCO_R_ELBOW,
        H36M_R_HAND:     COCO_R_WRIST,
        H36M_L_SHOULDER: COCO_L_SHOULDER,
        H36M_L_ELBOW:    COCO_L_ELBOW,
        H36M_L_HAND:     COCO_L_WRIST,
    }

    for h36m_idx, coco_idx in mapping.items():
        h36m_kp[h36m_idx]   = coco_kp[coco_idx]
        h36m_conf[h36m_idx] = coco_conf[coco_idx]

    # -----------------------------
    # 2. 계산 필요 관절 (5개)
    # -----------------------------

    # Pelvis = (L.hip + R.hip) / 2
    l_hip, r_hip = coco_kp[COCO_L_HIP], coco_kp[COCO_R_HIP]
    h36m_kp[H36M_PELVIS]   = (l_hip + r_hip) / 2
    h36m_conf[H36M_PELVIS] = (coco_conf[COCO_L_HIP] + coco_conf[COCO_R_HIP]) / 2

    # Thorax = (L.shoulder + R.shoulder) / 2
    l_sho, r_sho = coco_kp[COCO_L_SHOULDER], coco_kp[COCO_R_SHOULDER]
    h36m_kp[H36M_THORAX]   = (l_sho + r_sho) / 2
    h36m_conf[H36M_THORAX] = (coco_conf[COCO_L_SHOULDER] + coco_conf[COCO_R_SHOULDER]) / 2

    # Spine = (Pelvis + Thorax) / 2
    h36m_kp[H36M_SPINE]   = (h36m_kp[H36M_PELVIS] + h36m_kp[H36M_THORAX]) / 2
    h36m_conf[H36M_SPINE] = (h36m_conf[H36M_PELVIS] + h36m_conf[H36M_THORAX]) / 2

    # Neck base = Thorax 좌표 그대로 사용 (학계 표준)
    # 참고: 많은 논문에서 Human3.6M의 "Neck/Nose"를 버리고
    #       "Thorax"를 "Neck"으로 매핑함
    h36m_kp[H36M_NECK]   = h36m_kp[H36M_THORAX].copy()
    h36m_conf[H36M_NECK] = h36m_conf[H36M_THORAX]

    # Head = Neck에서 Nose 방향으로 "고정 비율" 연장
    # (Nose 좌표를 그대로 쓰면 고개를 들 때 head가
    #  앞으로 끌려가 숙인 것처럼 보이는 왜곡 발생)
    nose = coco_kp[COCO_NOSE]
    neck = h36m_kp[H36M_NECK]

    # 머리 길이를 어깨너비 기준으로 고정 (약 0.5배)
    shoulder_width = np.linalg.norm(l_sho - r_sho)
    head_length = shoulder_width * 0.5

    direction = nose - neck
    dir_norm = np.linalg.norm(direction)

    if dir_norm > 1e-6:
        unit_dir = direction / dir_norm
        h36m_kp[H36M_HEAD] = neck + unit_dir * head_length
    else:
        h36m_kp[H36M_HEAD] = nose

    h36m_conf[H36M_HEAD] = coco_conf[COCO_NOSE]

    return h36m_kp, h36m_conf


def convert_npz(npz_path, output_path):
    """
    NPZ 파일 전체(모든 프레임)를 COCO17 → H36M17로 변환 후 저장
    """
    data = np.load(npz_path)

    coco_keypoints  = data['keypoints']    # (N, 17, 2)
    coco_confidence = data['confidence']   # (N, 17)
    detected        = data['detected']     # (N,)
    fps             = data['fps']
    resolution      = data['resolution']
    frame_count     = data['frame_count']

    N = coco_keypoints.shape[0]

    h36m_keypoints  = np.zeros((N, 17, 2), dtype=np.float32)
    h36m_confidence = np.zeros((N, 17),    dtype=np.float32)

    for i in range(N):
        h36m_kp, h36m_conf = coco17_to_h36m17(coco_keypoints[i], coco_confidence[i])
        h36m_keypoints[i]  = h36m_kp
        h36m_confidence[i] = h36m_conf

    np.savez(
        output_path,
        keypoints   = h36m_keypoints,
        confidence  = h36m_confidence,
        detected    = detected,
        frame_count = frame_count,
        fps         = fps,
        resolution  = resolution
    )

    return N


# =============================================
# 메인: 7개 폴더 순회 변환
# =============================================
if __name__ == "__main__":
    print("="*60)
    print("🔄 COCO 17 → Human3.6M 17 관절 변환기")
    print("="*60)

    total_files   = 0
    success_files = 0
    failed_files  = []

    for folder_idx, folder_name in enumerate(FOLDERS, 1):
        folder_path     = os.path.join(BASE_PATH, folder_name)
        coco_path       = os.path.join(folder_path, "Keypoints_2D")
        h36m_path       = os.path.join(folder_path, "Keypoints_2D_H36M")

        print(f"\n[{folder_idx}/7] {folder_name}")
        print("-" * 50)

        if not os.path.exists(coco_path):
            print(f"  ❌ Keypoints_2D 폴더 없음")
            continue

        os.makedirs(h36m_path, exist_ok=True)

        npz_files = sorted(glob.glob(os.path.join(coco_path, "*.npz")))

        if not npz_files:
            print(f"  ❌ NPZ 파일 없음")
            continue

        print(f"  파일 수: {len(npz_files)}개")

        for f_idx, npz_path in enumerate(npz_files, 1):
            video_name  = os.path.splitext(os.path.basename(npz_path))[0]
            output_path = os.path.join(h36m_path, video_name)

            if os.path.exists(output_path + ".npz"):
                success_files += 1
                total_files += 1
                continue

            try:
                n_frames = convert_npz(npz_path, output_path)
                success_files += 1
            except Exception as e:
                print(f"  ❌ 실패: {video_name} - {e}")
                failed_files.append(npz_path)

            total_files += 1

            if f_idx % 100 == 0 or f_idx == len(npz_files):
                print(f"  진행: {f_idx}/{len(npz_files)}", end='\r')

        print(f"  ✓ 완료: {len(npz_files)}개")

    print(f"\n{'='*60}")
    print(f"✅ 전체 변환 완료!")
    print(f"{'='*60}")
    print(f"총 파일:  {total_files}개")
    print(f"성공:     {success_files}개")
    print(f"실패:     {len(failed_files)}개")

    if failed_files:
        print(f"\n❌ 실패한 파일:")
        for f in failed_files:
            print(f"   {f}")

    print(f"\n📁 저장 위치: 각 폴더의 Keypoints_2D_H36M/")
