"""
Generate side-by-side comparison videos for the project page.
Each output: [Source Video | Generated Video] with black bar labels on top.
"""
import cv2
import numpy as np
import os

BASE = "static/videos"
OUTPUT_FPS = 24
TARGET_FRAMES = 81
CONTENT_H = 480       # content height per video
BAR_H = 32            # black bar height for labels
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.7
FONT_THICK = 2
FONT_COLOR = (255, 255, 255)

# Folders and the 4 methods to compare (case-insensitive matching)
FOLDERS = [
    "comp_gen3c_batch_0005_rgb_cam_type3",
    "comp_gen3c_batch_0008_cam_type11",
    "comp_gen3c_batch_0009_rgb_cam_type11",
    "comp_rcm_1_cam_type13",
    "comp_rcm_4_cam_type13",
]
METHODS = ["InfCam", "GEN3C", "RCM", "TrajCrafter"]


def find_file(folder_path, name):
    """Case-insensitive file match."""
    for f in os.listdir(folder_path):
        if f.lower() == name.lower():
            return os.path.join(folder_path, f)
    return None


def read_video(path, target_frames):
    """Read video, return list of frames (BGR), capped at target_frames."""
    cap = cv2.VideoCapture(path)
    frames = []
    while len(frames) < target_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    # If video is shorter, loop last frame
    while len(frames) < target_frames:
        frames.append(frames[-1].copy())
    return frames[:target_frames]


def resize_to_height(frame, target_h):
    """Resize frame to target height, preserving aspect ratio."""
    h, w = frame.shape[:2]
    scale = target_h / h
    new_w = int(w * scale)
    return cv2.resize(frame, (new_w, target_h), interpolation=cv2.INTER_LANCZOS4)


def make_comparison_video(source_path, method_path, output_path, left_label, right_label):
    """Create side-by-side video with labels."""
    print(f"  Creating: {output_path}")
    src_frames = read_video(source_path, TARGET_FRAMES)
    met_frames = read_video(method_path, TARGET_FRAMES)

    # Resize both to CONTENT_H
    src_frames = [resize_to_height(f, CONTENT_H) for f in src_frames]
    met_frames = [resize_to_height(f, CONTENT_H) for f in met_frames]

    src_w = src_frames[0].shape[1]
    met_w = met_frames[0].shape[1]
    total_w = src_w + met_w
    total_h = CONTENT_H + BAR_H

    # Write with mp4v first, then re-encode to H.264 via ffmpeg for browser compatibility
    tmp_path = output_path.replace(".mp4", "_tmp.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(tmp_path, fourcc, OUTPUT_FPS, (total_w, total_h))

    for i in range(TARGET_FRAMES):
        canvas = np.zeros((total_h, total_w, 3), dtype=np.uint8)

        # Black bar with labels
        left_size = cv2.getTextSize(left_label, FONT, FONT_SCALE, FONT_THICK)[0]
        right_size = cv2.getTextSize(right_label, FONT, FONT_SCALE, FONT_THICK)[0]
        left_x = (src_w - left_size[0]) // 2
        right_x = src_w + (met_w - right_size[0]) // 2
        text_y = (BAR_H + left_size[1]) // 2
        cv2.putText(canvas, left_label, (left_x, text_y), FONT, FONT_SCALE, FONT_COLOR, FONT_THICK, cv2.LINE_AA)
        cv2.putText(canvas, right_label, (right_x, text_y), FONT, FONT_SCALE, FONT_COLOR, FONT_THICK, cv2.LINE_AA)

        # Video content
        canvas[BAR_H:, :src_w] = src_frames[i]
        canvas[BAR_H:, src_w:] = met_frames[i]

        writer.write(canvas)

    writer.release()

    # Re-encode to H.264 for browser playback
    import subprocess
    subprocess.run([
        "ffmpeg", "-y", "-i", tmp_path,
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-pix_fmt", "yuv420p", "-an", output_path,
    ], capture_output=True)
    os.remove(tmp_path)


def main():
    for folder in FOLDERS:
        folder_path = os.path.join(BASE, folder)
        source_path = find_file(folder_path, "source.mp4")
        if not source_path:
            print(f"SKIP {folder}: no source.mp4")
            continue

        print(f"\n=== {folder} ===")
        for method in METHODS:
            method_path = find_file(folder_path, f"{method}.mp4")
            if not method_path:
                print(f"  SKIP {method}: not found")
                continue

            out_name = f"{folder}_{method.lower()}.mp4"
            out_path = os.path.join(BASE, out_name)
            make_comparison_video(
                source_path, method_path, out_path,
                "Source Video", "Generated Video"
            )

    print("\nDone!")


if __name__ == "__main__":
    main()
