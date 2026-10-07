#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
YOLO26n-pose 기반 2D 관절 좌표 추출기
7개 폴더의 모든 비디오에 대해 COCO 17 관절 좌표를 NPZ 파일로 저장

저장 구조:
  RoseLab+Le2i/
  └── Coffee_room_01/
      ├── Videos/          ← 원본 비디오
      ├── Annotation_files/
      └── Keypoints_2D/    ← 새로 생성 (관절 좌표 저장)
          └── video (1).npz
          └── video (2).npz
          └── ...
"""

import cv2
import numpy as np
import os
import glob
import time
from ultralytics import YOLO

# =============================================
# 설정
# =============================================
BASE_PATH = r"C:\Users\User\OneDrive\data\RoseLab+Le2i+3AI"
MODEL_PATH = "yolo26x-pose.pt"

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

# COCO 17 관절 이름 (인덱스 참조용)
COCO_KEYPOINTS = [
    "Nose",           # 0
    "Left Eye",       # 1
    "Right Eye",      # 2
    "Left Ear",       # 3
    "Right Ear",      # 4
    "Left Shoulder",  # 5
    "Right Shoulder", # 6
    "Left Elbow",     # 7
    "Right Elbow",    # 8
    "Left Wrist",     # 9
    "Right Wrist",    # 10
    "Left Hip",       # 11
    "Right Hip",      # 12
    "Left Knee",      # 13
    "Right Knee",     # 14
    "Left Ankle",     # 15
    "Right Ankle"     # 16
]

# =============================================
# 모델 로드
# =============================================
print("="*60)
print("🚀 YOLO26n-pose 2D 관절 좌표 추출기")
print("="*60)
print(f"\n모델 로드 중: {MODEL_PATH}")

try:
    model = YOLO(MODEL_PATH, verbose=False)
    print("✓ 모델 로드 완료!\n")
except Exception as e:
    print(f"❌ 모델 로드 실패: {e}")
    print("yolo26n-pose.pt 파일이 같은 폴더에 있는지 확인하세요.")
    exit()

# =============================================
# 단일 비디오 처리 함수
# =============================================
def process_video(video_path, output_path):
    """
    단일 비디오에서 2D 관절 좌표 추출 후 NPZ로 저장
    
    저장 형식 (NPZ):
        keypoints:    (N, 17, 2)  - x, y 좌표
        confidence:   (N, 17)     - 각 관절 신뢰도
        detected:     (N,)        - 프레임별 감지 여부 (True/False)
        frame_count:  스칼라       - 총 프레임 수
        fps:          스칼라       - 비디오 FPS
        resolution:   (2,)        - [width, height]
    
    N = 총 프레임 수
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"    ❌ 비디오 열기 실패: {video_path}")
        return False

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 결과 저장 배열
    all_keypoints   = np.zeros((total_frames, 17, 2), dtype=np.float32)
    all_confidence  = np.zeros((total_frames, 17),    dtype=np.float32)
    all_detected    = np.zeros((total_frames,),        dtype=bool)

    frame_idx = 0
    detected_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # YOLO 추론
        results = model(frame, verbose=False)

        if (results and
            results[0].keypoints is not None and
            results[0].keypoints.xy.shape[0] > 0):

            kp = results[0].keypoints

            # 첫 번째 사람만 사용
            xy   = kp.xy[0].cpu().numpy()    # (17, 2)
            conf = kp.conf[0].cpu().numpy()   # (17,)

            all_keypoints[frame_idx]  = xy
            all_confidence[frame_idx] = conf
            all_detected[frame_idx]   = True
            detected_count += 1

        frame_idx += 1

        # 진행률
        progress = (frame_idx / total_frames) * 100
        print(f"    [{('='*int(progress/5)):<50}] {progress:.1f}%", end='\r')

    cap.release()

    # 실제 처리된 프레임 수로 자르기
    all_keypoints  = all_keypoints[:frame_idx]
    all_confidence = all_confidence[:frame_idx]
    all_detected   = all_detected[:frame_idx]

    # NPZ 저장
    np.savez(
        output_path,
        keypoints   = all_keypoints,
        confidence  = all_confidence,
        detected    = all_detected,
        frame_count = frame_idx,
        fps         = fps,
        resolution  = np.array([width, height])
    )

    detection_rate = detected_count / frame_idx * 100 if frame_idx > 0 else 0
    print(f"    [{('='*50)}] 100.0%")
    print(f"    ✓ 완료: {frame_idx}프레임, 감지율 {detection_rate:.1f}%")
    print(f"    저장: {output_path}.npz")

    return True


# =============================================
# 메인: 7개 폴더 순회
# =============================================
total_videos    = 0
success_videos  = 0
failed_videos   = []
start_total     = time.time()

for folder_idx, folder_name in enumerate(FOLDERS, 1):
    folder_path  = os.path.join(BASE_PATH, folder_name)
    videos_path  = os.path.join(folder_path, "Videos")
    keypoints_path = os.path.join(folder_path, "Keypoints_2D")

    print(f"\n[{folder_idx}/7] {folder_name}")
    print("-" * 50)

    # 폴더 존재 확인
    if not os.path.exists(videos_path):
        print(f"  ❌ Videos 폴더 없음: {videos_path}")
        continue

    # Keypoints_2D 폴더 생성
    os.makedirs(keypoints_path, exist_ok=True)
    print(f"  저장 폴더: {keypoints_path}")

    # 비디오 파일 목록
    video_files = sorted(
        glob.glob(os.path.join(videos_path, "*.avi")) +
        glob.glob(os.path.join(videos_path, "*.mp4"))
    )

    if not video_files:
        print(f"  ❌ 비디오 파일 없음")
        continue

    print(f"  비디오 수: {len(video_files)}개\n")

    for v_idx, video_path in enumerate(video_files, 1):
        video_name  = os.path.splitext(os.path.basename(video_path))[0]
        output_path = os.path.join(keypoints_path, video_name)

        # 이미 처리된 파일 스킵
        if os.path.exists(output_path + ".npz"):
            print(f"  [{v_idx}/{len(video_files)}] ⏭ 스킵 (이미 존재): {video_name}")
            success_videos += 1
            total_videos += 1
            continue

        print(f"  [{v_idx}/{len(video_files)}] 처리 중: {video_name}")

        start = time.time()
        success = process_video(video_path, output_path)
        elapsed = time.time() - start

        total_videos += 1
        if success:
            success_videos += 1
            print(f"    ⏱ 소요 시간: {elapsed:.1f}초")
        else:
            failed_videos.append(video_path)

# =============================================
# 최종 결과
# =============================================
total_elapsed = time.time() - start_total

print(f"\n{'='*60}")
print(f"✅ 전체 처리 완료!")
print(f"{'='*60}")
print(f"총 비디오:     {total_videos}개")
print(f"성공:          {success_videos}개")
print(f"실패:          {len(failed_videos)}개")
print(f"총 소요 시간:  {total_elapsed/60:.1f}분")

if failed_videos:
    print(f"\n❌ 실패한 파일:")
    for f in failed_videos:
        print(f"   {f}")

print(f"\n📁 저장 위치:")
for folder_name in FOLDERS:
    kp_path = os.path.join(BASE_PATH, folder_name, "Keypoints_2D")
    if os.path.exists(kp_path):
        count = len(glob.glob(os.path.join(kp_path, "*.npz")))
        print(f"   {folder_name}/Keypoints_2D/ → {count}개 파일")

print(f"\n💡 NPZ 파일 읽는 법:")
print(f"   data = np.load('video (1).npz')")
print(f"   keypoints  = data['keypoints']   # (N, 17, 2)")
print(f"   confidence = data['confidence']  # (N, 17)")
print(f"   detected   = data['detected']    # (N,)")
