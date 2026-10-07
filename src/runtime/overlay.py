from __future__ import annotations
from pathlib import Path
import cv2


def write_overlay_video(input_video, output_video, timeline, profile='research'):
    cap=cv2.VideoCapture(str(input_video))
    if not cap.isOpened(): raise ValueError(f'Cannot open video for overlay: {input_video}')
    fps=cap.get(cv2.CAP_PROP_FPS) or 30.0; w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc=cv2.VideoWriter_fourcc(*'mp4v')
    out=cv2.VideoWriter(str(output_video),fourcc,float(fps),(w,h))
    if not out.isOpened(): cap.release(); raise RuntimeError(f'Cannot create overlay video: {output_video}')
    idx=0
    try:
        while True:
            ok,frame=cap.read()
            if not ok: break
            row=timeline[idx] if idx < len(timeline) else {}
            lines=[f"Frame {idx}", f"TCN: {row.get('tcn_label') or 'WARMUP'}", f"DBN: {row.get('dbn_label') or 'WARMUP'}"]
            if profile=='research':
                lines.insert(2,f"STDS pre: {row.get('stds_pre_label') or 'WARMUP'}")
                lines.insert(3,f"STDS post: {row.get('stds_post_label') or 'WARMUP'}")
            y=28
            for text in lines:
                cv2.putText(frame,text,(12,y),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2,cv2.LINE_AA); y+=28
            out.write(frame); idx+=1
    finally:
        cap.release(); out.release()
    if idx == 0: raise ValueError('No decoded frames while writing overlay')
    return idx
