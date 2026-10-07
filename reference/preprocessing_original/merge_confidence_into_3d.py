import numpy as np
import tkinter as tk
from tkinter import filedialog


def load_dataset(path):
    d = np.load(path, allow_pickle=True)
    for key in ("X", "y", "video_name", "start_frame"):
        if key not in d.files:
            raise ValueError(f"'{key}' 키가 없습니다: {path}")
    return d["X"], d["y"], d["video_name"], d["start_frame"]


def main():
    root = tk.Tk()
    root.withdraw()

    path_3d = filedialog.askopenfilename(
        title="3D 통합 데이터셋 선택 (confidence 없는 버전, x,y,z)",
        filetypes=[("NumPy npz", "*.npz"), ("All files", "*.*")]
    )
    if not path_3d:
        print("파일 선택 안 됨")
        return

    path_2d = filedialog.askopenfilename(
        title="confidence 포함된 2D 통합 데이터셋 선택 (x,y,confidence)",
        filetypes=[("NumPy npz", "*.npz"), ("All files", "*.*")]
    )
    if not path_2d:
        print("파일 선택 안 됨")
        return

    save_path = filedialog.asksaveasfilename(
        title="confidence 추가된 3D 데이터셋 저장 위치",
        defaultextension=".npz",
        initialfile="fall_dataset_3D_with_conf.npz",
        filetypes=[("NumPy npz", "*.npz")]
    )
    if not save_path:
        print("저장 경로 선택 안 됨")
        return

    X3, y3, vid3, sf3 = load_dataset(path_3d)
    X2, y2, vid2, sf2 = load_dataset(path_2d)

    print(f"\n▶ 3D: {path_3d}  X={X3.shape}")
    print(f"▶ 2D: {path_2d}  X={X2.shape}")

    if X3.shape[-1] != 3:
        print(f"⚠️ 3D 파일의 마지막 차원이 3이 아닙니다: {X3.shape} (x,y,z 기대)")
    if X2.shape[-1] != 3:
        print(f"⚠️ 2D 파일의 마지막 차원이 3이 아닙니다: {X2.shape} (x,y,confidence 기대)")

    # 2D 데이터셋에서 (video_name, start_frame) → 인덱스 매핑 생성
    # (같은 CSV + 같은 stride로 만들었다면 3D와 2D의 클립 위치가 1:1로 대응한다고 가정)
    index_2d = {}
    for i in range(len(vid2)):
        key = (str(vid2[i]), int(sf2[i]))
        index_2d[key] = i

    N = len(vid3)
    T, J = X3.shape[1], X3.shape[2]
    conf_list = []
    unmatched = []
    label_mismatch = []

    for i in range(N):
        key = (str(vid3[i]), int(sf3[i]))
        j = index_2d.get(key)
        if j is None:
            unmatched.append(key)
            conf_list.append(np.zeros((T, J), dtype=np.float32))  # 못 찾으면 0으로 채움
            continue
        if int(y3[i]) != int(y2[j]):
            label_mismatch.append(key)
        conf_list.append(X2[j][:, :, 2].astype(np.float32))  # 2D의 3번째 채널 = confidence

    conf_arr = np.stack(conf_list)                         # (N, 64, 17)
    X_merged = np.concatenate([X3, conf_arr[..., None]], axis=-1)  # (N,64,17,4): x,y,z,confidence

    np.savez_compressed(save_path, X=X_merged, y=y3, video_name=vid3, start_frame=sf3)

    print("\n" + "─" * 60)
    print(f"매칭 성공: {N - len(unmatched)}/{N}")
    if unmatched:
        print(f"⚠️ 매칭 실패({len(unmatched)}개, confidence=0으로 채움) 앞 5개 예시: {unmatched[:5]}")
        print("   → 2D/3D를 서로 다른 CSV나 다른 stride 설정으로 만들었을 가능성이 있습니다.")
    if label_mismatch:
        print(f"⚠️ 라벨 불일치({len(label_mismatch)}개) 앞 5개 예시: {label_mismatch[:5]}")
        print("   → 같은 클립인데 정답이 다릅니다. 2D/3D가 서로 다른 버전의 CSV로 만들어졌을 수 있습니다.")
    print(f"\n🎉 저장 완료: {save_path}")
    print(f"X shape: {X_merged.shape}  (마지막 채널 순서: x, y, z, confidence)")


if __name__ == "__main__":
    main()
