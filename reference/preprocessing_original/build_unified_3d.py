# -*- coding: utf-8 -*-
"""
build_unified_3d.py — 모든 환경의 3D 스켈레톤 + 프레임별 라벨을 하나의 npz로 통합

사용법:
  python build_unified_3d.py

  1) 상위 폴더 선택 (RoseLab+Le2i+3AI 같은, 환경 폴더들이 들어있는 폴더)
  2) 저장 위치 선택

자동으로 하위 폴더를 순회하며:
  - {환경}/Keypoints_3D/*.npz  → 스켈레톤 (frames, 17, 3)
  - {환경}/Annotation_files/*.csv → 전조/위험 프레임 번호

출력 npz 구조 (프레임 단위 DB):
  skeletons     : (전체프레임수, 17, 3)  — 모든 영상의 프레임을 이어붙임
  labels        : (전체프레임수,)         — 프레임별 0=정상 / 1=전조 / 2=위험
  video_idx     : (전체프레임수,)         — 각 프레임이 몇 번째 영상인지
  video_names   : (영상수,)              — 영상 이름
  env_names     : (영상수,)              — 환경 이름
  video_starts  : (영상수,)              — skeletons에서 시작 인덱스
  video_lengths : (영상수,)              — 영상별 프레임 수

학습 시: 이 파일을 읽고, env_names로 train/test 분리 후,
         원하는 window/stride로 슬라이딩 → 클립 생성.
"""

import os, csv
import numpy as np
import tkinter as tk
from tkinter import filedialog

LABEL_NORMAL = 0
LABEL_PRECURSOR = 1
LABEL_DANGER = 2

NPZ_KEY_CANDIDATES = ["keypoints_3d", "skeletons", "skeleton", "keypoints", "pose", "data", "joints"]

# 통합에서 제외할 환경 (나중에 추가 예정인 데이터)
EXCLUDE_ENVS = {"RoseLab_S001"}


# ─── 유틸 ───
def is_empty(v):
    return v is None or str(v).strip() == ""

def parse_int(v, default=None):
    if is_empty(v):
        return None
    try:
        number = float(v)
        if not number.is_integer(): raise ValueError
        return int(number)
    except (TypeError, ValueError):
        raise ValueError(f"Invalid annotation frame value: {v!r}")

def read_csv_rows(path):
    for enc in ("utf-8-sig", "cp949", "utf-8"):
        try:
            with open(path, "r", newline="", encoding=enc) as f:
                return list(csv.reader(f))
        except UnicodeDecodeError:
            continue
    with open(path, "r", newline="", encoding="utf-8", errors="replace") as f:
        return list(csv.reader(f))

def get_sets_for_row(row, num_sets):
    sets = []
    for set_no in range(1, num_sets + 1):
        start_col = 1 + (set_no - 1) * 4
        vals = [row[start_col + j] if start_col + j < len(row) else "" for j in range(4)]
        if all(is_empty(v) for v in vals):
            continue
        p_s, p_e, d_s, d_e = (parse_int(v) for v in vals)
        if all(v is None for v in (p_s, p_e, d_s, d_e)):
            continue
        if p_s == 0 and p_e == 0 and d_s == 0 and d_e == 0:
            continue
        for label, a, b in (("precursor", p_s, p_e), ("danger", d_s, d_e)):
            if (a is None) != (b is None):
                raise ValueError(f"Incomplete {label} annotation interval: {(a, b)}")
            if a is not None and (a < 0 or b < a):
                raise ValueError(f"Invalid {label} annotation interval: {(a, b)}")
        sets.append((p_s, p_e, d_s, d_e))
    return sets

def build_frame_labels(total_frames, sets):
    labels = np.zeros(total_frames, dtype=np.int64)
    for (p_s, p_e, d_s, d_e) in sets:
        if p_s is not None and p_e is not None and p_e >= p_s:
            s, e = max(0, p_s), min(total_frames - 1, p_e)
            if e >= s:
                labels[s:e + 1] = LABEL_PRECURSOR
    for (p_s, p_e, d_s, d_e) in sets:
        if d_s is not None and d_e is not None and d_e >= d_s:
            s, e = max(0, d_s), min(total_frames - 1, d_e)
            if e >= s:
                labels[s:e + 1] = LABEL_DANGER
    return labels

def load_skeleton_3d(npz_path):
    """원본 3D 스켈레톤만 로드. 파생 데이터셋 파일이면 None 반환."""
    data = np.load(npz_path, allow_pickle=False)

    # 파생 데이터셋(fall_dataset_*.npz) 판별 → 제외
    # 원본은 'keypoints_3d' 키를 가짐. 'X','y' 키가 있으면 클립 데이터셋임.
    if "X" in data.files or "y" in data.files:
        return None

    key = None
    for cand in NPZ_KEY_CANDIDATES:
        if cand in data.files:
            key = cand; break
    if key is None:
        raise KeyError(f"지원되는 skeleton key를 찾을 수 없습니다. keys={data.files}")
    arr = data[key]
    if arr.ndim == 2 and arr.shape[1] == 17 * 3:
        arr = arr.reshape(-1, 17, 3)

    # 원본은 반드시 (frames, 17, 3) — 아니면 제외
    if arr.ndim != 3 or arr.shape[1] != 17:
        return None

    if arr.ndim != 3 or arr.shape[1:] != (17, 3) or not np.isfinite(arr).all():
        raise ValueError(f"Invalid 3D skeleton in {npz_path}: shape={arr.shape}")
    return arr.astype(np.float32)

def find_npz_file(folder_path, npz_map, video_name):
    name = video_name.strip()
    p = os.path.join(folder_path, name + ".npz")
    if os.path.isfile(p):
        return p
    key1 = (name + ".npz").lower()
    if key1 in npz_map:
        return os.path.join(folder_path, npz_map[key1])
    if name.lower() in npz_map:
        return os.path.join(folder_path, npz_map[name.lower()])
    return None


def find_env_folders(root_dir):
    """root_dir 아래에서 Keypoints_3D + Annotation_files 가 있는 폴더를 찾는다.
    EXCLUDE_ENVS 에 있는 환경은 제외."""
    envs = []
    for name in sorted(os.listdir(root_dir)):
        if name in EXCLUDE_ENVS:
            print(f"  (제외) {name}")
            continue
        sub = os.path.join(root_dir, name)
        if not os.path.isdir(sub):
            continue
        kp = os.path.join(sub, "Keypoints_3D")
        ann = os.path.join(sub, "Annotation_files")
        if os.path.isdir(kp) and os.path.isdir(ann):
            envs.append((name, kp, ann))
    return envs


def find_annotation_csv(env_name, ann_dir):
    """정확히 '{환경명}_annotations.csv' 파일만 선택.
    '- 복사본', '(1)' 등이 붙은 변형 파일은 제외한다."""
    target = f"{env_name}_annotations.csv".lower()
    for f in os.listdir(ann_dir):
        if f.lower() == target:
            return os.path.join(ann_dir, f)
    return None


def process_one_env(env_name, kp_dir, ann_dir):
    """한 환경의 모든 영상을 처리. (skeleton_list, label_list, name_list) 반환."""
    # CSV 찾기 — 정확히 {환경명}_annotations.csv 만
    csv_path_main = find_annotation_csv(env_name, ann_dir)
    if csv_path_main is None:
        print(f"  ⚠️ {env_name}: '{env_name}_annotations.csv' 없음 → 건너뜀")
        return [], [], []
    print(f"  CSV: {os.path.basename(csv_path_main)}")
    csv_files = [os.path.basename(csv_path_main)]

    # NPZ 맵
    npz_files = [f for f in os.listdir(kp_dir) if f.lower().endswith(".npz")]
    npz_map = {f.lower(): f for f in npz_files}

    skeletons, labels, names = [], [], []

    for csv_file in csv_files:
        csv_path = os.path.join(ann_dir, csv_file)
        rows = read_csv_rows(csv_path)
        if len(rows) < 2:
            continue
        header = rows[0]
        num_sets = max(1, (len(header) - 1) // 4)

        for row in rows[1:]:
            if len(row) == 0 or is_empty(row[0]):
                continue
            video_name = row[0].strip()

            npz_path = find_npz_file(kp_dir, npz_map, video_name)
            if npz_path is None:
                continue

            skel = load_skeleton_3d(npz_path)
            if skel is None:
                print(f"    ⚠️ {video_name}: 원본 스켈레톤 형식 아님 → 건너뜀")
                continue
            total_frames = skel.shape[0]
            if total_frames < 2:
                continue

            sets = get_sets_for_row(row, num_sets)
            fl = build_frame_labels(total_frames, sets)

            skeletons.append(skel)
            labels.append(fl)
            names.append(video_name)

    # CSV에 없는 npz = 정상 영상 (라벨 전부 0)
    csv_videos = set(n.lower() for n in names)
    for npz_file in npz_files:
        vname = os.path.splitext(npz_file)[0]
        if vname.lower() in csv_videos:
            continue
        skel = load_skeleton_3d(os.path.join(kp_dir, npz_file))
        if skel is None:
            continue
        if skel.shape[0] < 2:
            continue
        fl = np.zeros(skel.shape[0], dtype=np.int64)
        skeletons.append(skel)
        labels.append(fl)
        names.append(vname)

    return skeletons, labels, names


# ─── 메인 ───
def main():
    root = tk.Tk()
    root.withdraw()

    # 1) 상위 폴더 선택
    root_dir = filedialog.askdirectory(
        title="상위 폴더 선택 (환경 폴더들이 들어있는 폴더)")
    if not root_dir:
        print("폴더 선택 안 됨"); return

    # 2) 저장 위치
    save_path = filedialog.asksaveasfilename(
        title="통합 데이터 저장 위치",
        defaultextension=".npz",
        initialfile="unified_3d_all_envs.npz",
        filetypes=[("NumPy npz", "*.npz")])
    if not save_path:
        print("저장 경로 선택 안 됨"); return

    # 환경 폴더 탐색
    envs = find_env_folders(root_dir)
    if not envs:
        print("환경 폴더를 찾을 수 없습니다."); return

    print(f"\n{'='*60}")
    print(f"  통합 3D 데이터셋 빌더 (프레임 단위)")
    print(f"{'='*60}")
    print(f"  상위 폴더: {root_dir}")
    print(f"  발견된 환경: {[e[0] for e in envs]}")
    print(f"{'='*60}\n")

    all_skeletons = []   # 영상별 (T_i, 17, 3)
    all_labels = []      # 영상별 (T_i,)
    all_vnames = []      # 영상 이름
    all_enames = []      # 환경 이름

    for env_name, kp_dir, ann_dir in envs:
        print(f"▶ {env_name}")
        skels, labs, names = process_one_env(env_name, kp_dir, ann_dir)
        print(f"  → {len(names)}개 영상, "
              f"총 {sum(s.shape[0] for s in skels)}프레임")

        all_skeletons.extend(skels)
        all_labels.extend(labs)
        all_vnames.extend(names)
        all_enames.extend([env_name] * len(names))

    if not all_skeletons:
        print("처리된 영상 없음"); return

    # 이어붙이기
    concat_skel = np.concatenate(all_skeletons, axis=0)   # (전체, 17, 3)
    concat_labels = np.concatenate(all_labels, axis=0)     # (전체,)

    # 영상별 메타데이터
    lengths = np.array([s.shape[0] for s in all_skeletons], dtype=np.int64)
    starts = np.cumsum(lengths) - lengths  # 시작 인덱스
    video_idx = np.concatenate([
        np.full(l, i, dtype=np.int64) for i, l in enumerate(lengths)
    ])

    # 저장
    np.savez_compressed(
        save_path,
        skeletons=concat_skel,
        labels=concat_labels,
        video_idx=video_idx,
        video_names=np.array(all_vnames),
        env_names=np.array(all_enames),
        video_starts=starts,
        video_lengths=lengths,
    )

    total = concat_skel.shape[0]
    n_videos = len(all_vnames)
    n_normal = int((concat_labels == 0).sum())
    n_precur = int((concat_labels == 1).sum())
    n_danger = int((concat_labels == 2).sum())

    print(f"\n{'='*60}")
    print(f"  ✅ 저장 완료: {save_path}")
    print(f"{'='*60}")
    print(f"  영상 수  : {n_videos}")
    print(f"  총 프레임: {total:,}")
    print(f"  skeletons: {concat_skel.shape}")
    print(f"  labels   : {concat_labels.shape}")
    print(f"\n  프레임별 라벨 분포:")
    print(f"    정상(0): {n_normal:,} ({n_normal/total*100:.1f}%)")
    print(f"    전조(1): {n_precur:,} ({n_precur/total*100:.1f}%)")
    print(f"    위험(2): {n_danger:,} ({n_danger/total*100:.1f}%)")
    print(f"\n  환경별 영상 수:")
    for env_name, _, _ in envs:
        cnt = sum(1 for e in all_enames if e == env_name)
        frames = sum(s.shape[0] for s, e in zip(all_skeletons, all_enames) if e == env_name)
        print(f"    {env_name}: {cnt}영상, {frames:,}프레임")


if __name__ == "__main__":
    main()
