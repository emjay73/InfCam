"""
Split 2x3 grid comparison videos into per-method [Source | Generated] videos.
Grid layout:
  Row 0: Source Video | ReCamMaster | ReCamMaster w/ interp
  Row 1: GEN3C        | TrajectoryCrafter | Ours (InfCam)
"""
import cv2
import numpy as np
import os
import subprocess

OUT_DIR = "static/videos"
BAR_H = 32
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.7
FONT_THICK = 2
FONT_COLOR = (255, 255, 255)

# Grid: 1920x1080, 3 cols x 2 rows
CELL_W = 640
CELL_H = 540

# Cell positions: (row, col)
CELL_MAP = {
    "source": (0, 0),
    "ReCamMaster": (0, 1),
    "ReCamMaster_interp": (0, 2),
    "GEN3C": (1, 0),
    "TrajectoryCrafter": (1, 1),
    "InfCam": (1, 2),
}

# Method order
METHODS = ["InfCam", "TrajectoryCrafter", "GEN3C", "ReCamMaster", "ReCamMaster_interp"]

# Input videos
INPUTS = [
    "static/videos/comparison_webvid_1.mp4",
    "static/videos/comparison_webvid_2.mp4",
    "static/videos/comparison_webvid_3.mp4",
    "static/videos/comparison_in-the-wild_1.mp4",
    "static/videos/comparison_in-the-wild_2.mp4",
]


def extract_cell(frame, row, col):
    y = row * CELL_H
    x = col * CELL_W
    return frame[y:y + CELL_H, x:x + CELL_W]


# Known content sizes per row (measured from grid videos)
# Row 0 (Source, RCM, RCM w/interp): full 640x540
# Row 1 (GEN3C, TC, Ours): 640x360 (bottom 180px is black padding)
def auto_crop(cell):
    """Remove black padding from top and bottom by detecting content rows."""
    gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
    h = gray.shape[0]
    top = 0
    for y in range(h):
        if gray[y, :].mean() > 5:
            top = y
            break
    bottom = h
    for y in range(h - 1, -1, -1):
        if gray[y, :].mean() > 5:
            bottom = y + 1
            break
    return cell[top:bottom, :]


def make_video(frames, fps, method, out_path):
    src_pos = CELL_MAP["source"]
    met_pos = CELL_MAP[method]

    # Detect content height from a bright frame (skip black intro frames)
    detect_frame = frames[0]
    for f in frames:
        cell = extract_cell(f, *src_pos)
        if cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY).mean() > 5:
            detect_frame = f
            break
    src0 = auto_crop(extract_cell(detect_frame, *src_pos))
    met0 = auto_crop(extract_cell(detect_frame, *met_pos))
    content_h = min(src0.shape[0], met0.shape[0])

    target_h = 480
    target_cw = int(CELL_W * target_h / content_h)

    total_w = target_cw * 2
    total_h = target_h + BAR_H

    tmp_path = out_path.replace(".mp4", "_tmp.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(tmp_path, fourcc, fps, (total_w, total_h))

    labels = ["Source Video", "Generated Video"]

    for frame in frames:
        src = auto_crop(extract_cell(frame, *src_pos))
        met = auto_crop(extract_cell(frame, *met_pos))

        # Ensure same height
        src = src[:content_h, :]
        met = met[:content_h, :]

        src = cv2.resize(src, (target_cw, target_h), interpolation=cv2.INTER_LANCZOS4)
        met = cv2.resize(met, (target_cw, target_h), interpolation=cv2.INTER_LANCZOS4)

        canvas = np.zeros((total_h, total_w, 3), dtype=np.uint8)

        for lbl, offset in zip(labels, [0, target_cw]):
            sz = cv2.getTextSize(lbl, FONT, FONT_SCALE, FONT_THICK)[0]
            lx = offset + (target_cw - sz[0]) // 2
            ly = (BAR_H + sz[1]) // 2
            cv2.putText(canvas, lbl, (lx, ly), FONT, FONT_SCALE, FONT_COLOR, FONT_THICK, cv2.LINE_AA)

        canvas[BAR_H:, :target_cw] = src
        canvas[BAR_H:, target_cw:] = met

        writer.write(canvas)

    writer.release()

    subprocess.run([
        "ffmpeg", "-y", "-i", tmp_path,
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-pix_fmt", "yuv420p", "-an", out_path,
    ], capture_output=True)
    os.remove(tmp_path)


def main():
    for src_path in INPUTS:
        basename = os.path.basename(src_path).replace(".mp4", "")
        # e.g. comparison_webvid_1 -> webvid1, comparison_in-the-wild_1 -> itw1
        if "webvid" in basename:
            tag = "webvid" + basename.split("_")[-1]
        else:
            tag = "itw" + basename.split("_")[-1]

        print(f"\n=== {basename} -> {tag} ===")

        cap = cv2.VideoCapture(src_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()

        # Remove leading black frames
        while frames and cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY).mean() < 3:
            frames.pop(0)
        print(f"  Read {len(frames)} frames (after removing black intro), fps={fps}")

        for method in METHODS:
            out_name = f"realworld_{tag}_{method.lower()}.mp4"
            out_path = os.path.join(OUT_DIR, out_name)
            make_video(frames, fps, method, out_path)
            print(f"  OK: {out_name}")

    print("\nDone!")


if __name__ == "__main__":
    main()
