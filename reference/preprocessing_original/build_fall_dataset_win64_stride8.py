# -*- coding: utf-8 -*-
"""
build_fall_dataset_win64_stride8.py — 하이브리드 모델용 64프레임 데이터셋 빌더 (3D + 2D 통합)

사용법:
  python build_fall_dataset_win64_stride8.py

  실행하면 tkinter 대화상자로:
    1) CSV 파일 선택 (전조/낙상 라벨링 결과)
    2) 3D npz 폴더 선택  (x,y,z)
    3) 2D npz 폴더 선택  (x,y + confidence)
    4) 저장 폴더 선택

  저장 파일명 자동 생성:
    fall_dataset_3D_win64_stride8_{폴더이름}.npz
    fall_dataset_2D_confidence_win64_stride8_{폴더이름}.npz

npz 내용:
  X          : (N, 64, 17, C)  C=3
  y          : (N,)            0=정상 / 1=전조 / 2=위험
  video_name : (N,)            출처 영상 이름
  start_frame: (N,)            원본 영상에서의 시작 프레임
"""

import os
import csv
import numpy as np
import tkinter as tk
from tkinter import filedialog

# ─────────────────────────────────────────────
# 설정
# ─────────────────────────────────────────────
WINDOW = 64
STRIDE = 8
LABEL_NORMAL = 0
LABEL_PRECURSOR = 1
LABEL_DANGER = 2

NPZ_KEY_CANDIDATES = ["keypoints_3d", "skeletons", "skeleton", "keypoints", "pose", "data", "joints"]


# ─────────────────────────────────────────────
# 유틸
# ─────────────────────────────────────────────
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
    """CSV 한 행에서 (전조S, 전조E, 낙상S, 낙상E) 세트들을 뽑아낸다."""
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
    """프레임 라벨 배열 생성. 전조 먼저 칠하고 낙상이 덮어쓴다."""
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


def window_label(labels_slice):
    """윈도우의 마지막 프레임 라벨 = 해당 윈도우의 정답."""
    return int(labels_slice[-1])


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


def load_skeleton_3d(npz_path):
    """3D(x,y,z) 스켈레톤 로더. confidence 없이 그대로."""
    data = np.load(npz_path, allow_pickle=False)
    key = None
    for cand in NPZ_KEY_CANDIDATES:
        if cand in data.files:
            key = cand; break
    if key is None:
        raise KeyError(f"지원되는 skeleton key를 찾을 수 없습니다. keys={data.files}")
    arr = data[key]
    if arr.ndim == 2 and arr.shape[1] == 17 * 3:
        arr = arr.reshape(-1, 17, 3)
    if arr.ndim != 3 or arr.shape[1:] != (17, 3) or not np.isfinite(arr).all():
        raise ValueError(f"Invalid 3D skeleton in {npz_path}: shape={arr.shape}")
    return arr.astype(np.float32)


def load_skeleton_2d(npz_path):
    """2D(x,y) + confidence 스켈레톤 로더. confidence가 있으면 3번째 채널로 합친다."""
    data = np.load(npz_path, allow_pickle=False)
    key = None
    for cand in NPZ_KEY_CANDIDATES:
        if cand in data.files:
            key = cand; break
    if key is None:
        raise KeyError(f"지원되는 skeleton key를 찾을 수 없습니다. keys={data.files}")
    arr = data[key]
    if arr.ndim == 2 and arr.shape[1] == 17 * 3:
        arr = arr.reshape(-1, 17, 3)

    # confidence 병합
    if arr.ndim == 3 and arr.shape[-1] == 2 and "confidence" in data.files:
        conf = data["confidence"]
        if conf.shape == arr.shape[:2]:
            arr = np.concatenate([arr, conf[..., None]], axis=-1)
        else:
            print(f"  ⚠️ {os.path.basename(npz_path)}: confidence shape 불일치 → 병합 건너뜀")

    if arr.ndim != 3 or arr.shape[1:] != (17, 3) or not np.isfinite(arr).all():
        raise ValueError(f"Invalid 3D skeleton in {npz_path}: shape={arr.shape}")
    return arr.astype(np.float32)


def build_one(csv_rows, num_sets, npz_folder, npz_map, loader_fn, mode_label):
    """하나의 모드(3D 또는 2D)에 대해 윈도우 클립 생성."""
    X_list, y_list, video_list, start_list = [], [], [], []
    skipped = []

    for row in csv_rows:
        if len(row) == 0 or is_empty(row[0]):
            continue
        video_name = row[0].strip()

        npz_path = find_npz_file(npz_folder, npz_map, video_name)
        if npz_path is None:
            skipped.append(video_name)
            continue

        sets = get_sets_for_row(row, num_sets)
        skeleton = loader_fn(npz_path)
        total_frames = skeleton.shape[0]

        if total_frames < WINDOW:
            print(f"  ⚠️ [{mode_label}] {video_name}: {total_frames}프레임 < {WINDOW} → 건너뜀")
            skipped.append(video_name)
            continue

        labels = build_frame_labels(total_frames, sets)

        n_win = 0
        for start in range(0, total_frames - WINDOW + 1, STRIDE):
            end = start + WINDOW
            clip = skeleton[start:end]
            lbl = window_label(labels[start:end])
            X_list.append(clip)
            y_list.append(lbl)
            video_list.append(video_name)
            start_list.append(start)
            n_win += 1

    if not X_list:
        return None, None, None, None, skipped

    X = np.stack(X_list).astype(np.float32)
    y = np.array(y_list, dtype=np.int64)
    vid = np.array(video_list)
    sf = np.array(start_list, dtype=np.int64)
    return X, y, vid, sf, skipped


# ─────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────
def main():
    root = tk.Tk()
    root.withdraw()

    # 1) CSV 선택
    csv_path = filedialog.askopenfilename(
        title="CSV 파일 선택 (전조/낙상 라벨링 결과)",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
    )
    if not csv_path:
        print("CSV 선택 안 됨"); return

    # 2) 3D npz 폴더
    npz_3d_folder = filedialog.askdirectory(title="3D(x,y,z) npz 폴더 선택")
    if not npz_3d_folder:
        print("3D 폴더 선택 안 됨"); return

    # 3) 2D npz 폴더
    npz_2d_folder = filedialog.askdirectory(title="2D(x,y,confidence) npz 폴더 선택")
    if not npz_2d_folder:
        print("2D 폴더 선택 안 됨"); return

    # 4) 저장 폴더
    save_folder = filedialog.askdirectory(title="결과 저장 폴더 선택")
    if not save_folder:
        print("저장 폴더 선택 안 됨"); return

    # ── CSV 파싱 ──
    rows = read_csv_rows(csv_path)
    if len(rows) < 2:
        print("CSV에 데이터가 없습니다."); return
    header = rows[0]
    num_sets = max(1, (len(header) - 1) // 4)
    data_rows = rows[1:]

    # ── 폴더 이름 추출 (저장 파일명에 붙일 용도) ── 저장 폴더 이름 사용
    folder_tag = os.path.basename(os.path.normpath(save_folder))

    print(f"\n{'='*60}")
    print(f"  하이브리드용 데이터셋 빌더 (win{WINDOW}, stride{STRIDE})")
    print(f"{'='*60}")
    print(f"  CSV      : {csv_path}")
    print(f"  3D 폴더  : {npz_3d_folder}")
    print(f"  2D 폴더  : {npz_2d_folder}")
    print(f"  저장 폴더: {save_folder}")
    print(f"  폴더 태그: {folder_tag}")
    print(f"  세트 수  : {num_sets}")
    print(f"{'='*60}")

    # ── 3D 빌드 ──
    print(f"\n▶ [3D] 빌드 시작...")
    npz_3d_map = {f.lower(): f for f in os.listdir(npz_3d_folder) if f.lower().endswith(".npz")}
    X3, y3, vid3, sf3, skip3 = build_one(data_rows, num_sets, npz_3d_folder, npz_3d_map,
                                          load_skeleton_3d, "3D")
    if X3 is not None:
        fname_3d = f"fall_dataset_3D_win{WINDOW}_stride{STRIDE}_{folder_tag}.npz"
        path_3d = os.path.join(save_folder, fname_3d)
        np.savez_compressed(path_3d, X=X3, y=y3, video_name=vid3, start_frame=sf3)
        print(f"\n  ✅ 3D 저장: {path_3d}")
        print(f"     X: {X3.shape}, y: {y3.shape}")
        for cls, name in [(0, "정상"), (1, "전조"), (2, "위험")]:
            print(f"     · {name}({cls}): {int((y3 == cls).sum())}개")
    else:
        print("  ❌ 3D 클립 없음")
    if skip3:
        print(f"  건너뜀({len(skip3)}): {skip3}")

    # ── 2D 빌드 ──
    print(f"\n▶ [2D] 빌드 시작...")
    npz_2d_map = {f.lower(): f for f in os.listdir(npz_2d_folder) if f.lower().endswith(".npz")}
    X2, y2, vid2, sf2, skip2 = build_one(data_rows, num_sets, npz_2d_folder, npz_2d_map,
                                          load_skeleton_2d, "2D")
    if X2 is not None:
        fname_2d = f"fall_dataset_2D_confidence_win{WINDOW}_stride{STRIDE}_{folder_tag}.npz"
        path_2d = os.path.join(save_folder, fname_2d)
        np.savez_compressed(path_2d, X=X2, y=y2, video_name=vid2, start_frame=sf2)
        print(f"\n  ✅ 2D 저장: {path_2d}")
        print(f"     X: {X2.shape}, y: {y2.shape}")
        for cls, name in [(0, "정상"), (1, "전조"), (2, "위험")]:
            print(f"     · {name}({cls}): {int((y2 == cls).sum())}개")
    else:
        print("  ❌ 2D 클립 없음")
    if skip2:
        print(f"  건너뜀({len(skip2)}): {skip2}")

    print(f"\n{'='*60}")
    print(f"  완료!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
