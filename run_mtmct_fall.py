"""Edit paths below, then run: python run_mtmct_fall.py"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
# Must already be synchronized. No frame skipping, reuse, offset, or drift correction.
VIDEO1 = ROOT / 'videos/color7-415.mp4'
VIDEO2 = ROOT / 'videos/color7-435.mp4'
YOLO_WEIGHTS = ROOT / 'assets/yolo/yolo26x-pose.pt'
REID_WEIGHTS = ROOT / 'weights/osnet_x1_0_msmt17.pth'
OUTPUT_ROOT = ROOT / 'outputs/mtmc_fall'
DEVICE = 'cpu'
IMGSZ = 640
WRITE_VIDEO = True

if __name__ == '__main__':
    sys.path.insert(0, str(ROOT / 'src'))
    from mtmc_fall.pipeline import OfflineConfig, run_offline
    print(run_offline(OfflineConfig(ROOT, (VIDEO1, VIDEO2), YOLO_WEIGHTS, REID_WEIGHTS,
                                   OUTPUT_ROOT, DEVICE, IMGSZ, WRITE_VIDEO)))
