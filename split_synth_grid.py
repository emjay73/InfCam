"""
Split the 6x4 grid comparison video into per-example, per-method videos.
Layout: [Source Video | Generated Video | Ground Truth]
"""
import cv2
import numpy as np
import os
import subprocess

SRC = "static/videos/comparison_augmcv_testset.mp4"
OUT_DIR = "static/videos"
TARGET_FRAMES = 81
BAR_H = 32
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.55
FONT_THICK = 2
FONT_COLOR = (255, 255, 255)

# Grid layout
LABEL_W = 100
CELL_W = 640
CELL_H = 360
# Rows: 0=Source, 1=GEN3C, 2=RCM, 3=TC, 4=Ours, 5=GT
ROW_MAP = {
    "source": 0,
    "GEN3C": 1,
    "ReCamMaster": 2,
    "TrajectoryCrafter": 3,
    "InfCam": 4,
    "gt": 5,
}
# Method order: Ours > TC > GEN3C > RCM
METHODS = ["InfCam", "TrajectoryCrafter", "GEN3C", "ReCamMaster"]


def extract_cell(frame, row, col):
    y = row * CELL_H
    x = LABEL_W + col * CELL_W
    return frame[y:y + CELL_H, x:x + CELL_W]


def main():
    cap = cv2.VideoCapture(SRC)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    while len(frames) < TARGET_FRAMES:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    print(f"Read {len(frames)} frames, fps={fps}")

    labels_text = ["Source Video", "Generated Video", "Ground Truth"]

    for col in range(4):
        print(f"\n=== Example {col + 1} ===")
        for method in METHODS:
            src_row = ROW_MAP["source"]
            met_row = ROW_MAP[method]
            gt_row = ROW_MAP["gt"]

            cw = CELL_W
            total_w = cw * 3
            total_h = CELL_H + BAR_H

            out_name = f"synth_ex{col + 1}_{method.lower()}.mp4"
            out_path = os.path.join(OUT_DIR, out_name)
            tmp_path = out_path.replace(".mp4", "_tmp.mp4")

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(tmp_path, fourcc, fps, (total_w, total_h))

            for frame in frames:
                src = extract_cell(frame, src_row, col)
                met = extract_cell(frame, met_row, col)
                gt = extract_cell(frame, gt_row, col)

                canvas = np.zeros((total_h, total_w, 3), dtype=np.uint8)

                # Labels on black bar
                for lbl, offset in zip(labels_text, [0, cw, cw * 2]):
                    sz = cv2.getTextSize(lbl, FONT, FONT_SCALE, FONT_THICK)[0]
                    lx = offset + (cw - sz[0]) // 2
                    ly = (BAR_H + sz[1]) // 2
                    cv2.putText(canvas, lbl, (lx, ly), FONT, FONT_SCALE, FONT_COLOR, FONT_THICK, cv2.LINE_AA)

                canvas[BAR_H:, :cw] = src
                canvas[BAR_H:, cw:cw * 2] = met
                canvas[BAR_H:, cw * 2:] = gt

                writer.write(canvas)

            writer.release()

            # H.264 re-encode
            subprocess.run([
                "ffmpeg", "-y", "-i", tmp_path,
                "-c:v", "libx264", "-preset", "fast", "-crf", "23",
                "-pix_fmt", "yuv420p", "-an", out_path,
            ], capture_output=True)
            os.remove(tmp_path)
            print(f"  OK: {out_name}")

    print("\nDone!")


if __name__ == "__main__":
    main()
