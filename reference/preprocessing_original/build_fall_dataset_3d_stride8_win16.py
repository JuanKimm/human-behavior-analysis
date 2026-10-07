import os
import csv
import numpy as np
import tkinter as tk
from tkinter import filedialog

# ─────────────────────────────────────────────
# 설정
# ─────────────────────────────────────────────
WINDOW = 16          # 클립 길이(프레임) — 16프레임 클립 (TCN 16f 학습용)
STRIDE = 8           # 슬라이딩 윈도우 stride
LABEL_NORMAL = 0
LABEL_PRECURSOR = 1
LABEL_DANGER = 2

# npz 안에서 스켈레톤 배열을 찾을 때 시도해볼 키 이름들 (우선순위 순)
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
    """CSV 한 행에서 (전조S, 전조E, 낙상S, 낙상E) 세트들을 뽑아낸다.
    - 4칸이 전부 비어있으면: 그 세트 자체가 없는 것 → 스킵
    - 4칸이 전부 0이면: '이벤트 없음(정상 비디오)' 표시 → 스킵
    """
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
    """전체 프레임 길이의 라벨 배열(0=정상)을 만들고, 세트별 전조/낙상 구간을 표시한다.
    전조를 먼저 칠하고 낙상을 나중에 덮어써서, 혹시 겹치면 낙상(더 위험한 쪽)이 우선한다."""
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
    """윈도우(16프레임)의 정답은 마지막 프레임의 라벨로 정한다.
    (실시간/스트리밍 상황에서 최근 16프레임을 보고 '현재 상태'를 판정하는 것과 동일한 방식)"""
    return int(labels_slice[-1])


def load_skeleton(npz_path):
    """3D(x,y,z) 스켈레톤 전용 로더. confidence 병합 없이 그대로 로드한다."""
    data = np.load(npz_path, allow_pickle=False)
    key = None
    for cand in NPZ_KEY_CANDIDATES:
        if cand in data.files:
            key = cand
            break
    if key is None:
        raise KeyError(f"지원되는 skeleton key를 찾을 수 없습니다. keys={data.files}")
    arr = data[key]
    if arr.ndim == 2 and arr.shape[1] == 17 * 3:   # (frames, 51) 형태면 reshape
        arr = arr.reshape(-1, 17, 3)
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
    if name.lower() in npz_map:  # 이름에 이미 .npz까지 포함해서 준 경우 대비
        return os.path.join(folder_path, npz_map[name.lower()])
    return None


# ─────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────
def main():
    root = tk.Tk()
    root.withdraw()

    # 1) CSV 파일 선택 (전조/낙상 라벨링 결과)
    csv_path = filedialog.askopenfilename(
        title="CSV 파일 선택 (전조/낙상 라벨링 결과)",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
    )
    if not csv_path:
        print("CSV 파일 선택 안 됨")
        return

    # 2) npz 폴더 선택
    npz_folder = filedialog.askdirectory(title="3D(x,y,z) npz 스켈레톤 파일들이 있는 폴더 선택")
    if not npz_folder:
        print("폴더 선택 안 됨")
        return

    # 3) 결과 저장 위치
    save_path = filedialog.asksaveasfilename(
        title="통합 데이터셋 저장 위치 (3D, confidence 없음)",
        defaultextension=".npz",
        initialfile="fall_dataset_3D_win16_stride8.npz",
        filetypes=[("NumPy npz", "*.npz")]
    )
    if not save_path:
        print("저장 경로 선택 안 됨")
        return

    rows = read_csv_rows(csv_path)
    if len(rows) < 2:
        print("CSV에 데이터가 없습니다.")
        return

    header = rows[0]
    num_sets = max(1, (len(header) - 1) // 4)

    npz_files = [f for f in os.listdir(npz_folder) if f.lower().endswith(".npz")]
    npz_map = {f.lower(): f for f in npz_files}

    X_list, y_list, video_list, start_list = [], [], [], []
    skipped_videos = []

    print(f"\n▶ CSV : {csv_path}")
    print(f"▶ npz 폴더: {npz_folder}")
    print(f"▶ 세트 수(헤더 기준): {num_sets}")
    print(f"▶ 윈도우={WINDOW}프레임, stride={STRIDE}")
    print("─" * 60)

    for row in rows[1:]:
        if len(row) == 0 or is_empty(row[0]):
            continue
        video_name = row[0].strip()

        npz_path = find_npz_file(npz_folder, npz_map, video_name)
        if npz_path is None:
            print(f"⚠️ npz 파일을 찾을 수 없음: '{video_name}' → 건너뜀")
            skipped_videos.append(video_name)
            continue

        sets = get_sets_for_row(row, num_sets)
        skeleton = load_skeleton(npz_path)
        total_frames = skeleton.shape[0]

        if total_frames < WINDOW:
            print(f"⚠️ {video_name}: 프레임 수({total_frames}) < {WINDOW} → 건너뜀")
            skipped_videos.append(video_name)
            continue

        labels = build_frame_labels(total_frames, sets)

        n_windows = 0
        for start in range(0, total_frames - WINDOW + 1, STRIDE):
            end = start + WINDOW
            clip = skeleton[start:end]          # (16, 17, C)
            lbl = window_label(labels[start:end])

            X_list.append(clip)
            y_list.append(lbl)
            video_list.append(video_name)
            start_list.append(start)
            n_windows += 1

        print(f"✅ {video_name}: 총 {total_frames}프레임 → {n_windows}개 클립 생성")

    if not X_list:
        print("생성된 클립이 없습니다.")
        return

    X = np.stack(X_list).astype(np.float32)   # (N, 64, 17, 3)
    y = np.array(y_list, dtype=np.int64)       # (N,)
    video_arr = np.array(video_list)
    start_arr = np.array(start_list, dtype=np.int64)

    np.savez_compressed(
        save_path,
        X=X, y=y, video_name=video_arr, start_frame=start_arr
    )

    print("\n" + "─" * 60)
    print(f"🎉 저장 완료: {save_path}")
    print(f"X shape: {X.shape}, y shape: {y.shape}")
    for cls, name in [(LABEL_NORMAL, "정상"), (LABEL_PRECURSOR, "전조"), (LABEL_DANGER, "낙상")]:
        cnt = int((y == cls).sum())
        print(f"  · {name}({cls}): {cnt}개")
    if skipped_videos:
        print(f"건너뛴 비디오({len(skipped_videos)}개): {skipped_videos}")


if __name__ == "__main__":
    main()
