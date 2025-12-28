import os
import random
import cv2
import subprocess
from pathlib import Path
import json
import shutil
import argparse

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

def main(args):

    input_dir = args.input_dir + "/videos" 
    tmp_dir = args.output_dir+"/tmp" 
    final_dir = args.output_dir+"/videos" 
    camera_dir = args.output_dir+"/cameras"

    # directory setup
    os.makedirs(tmp_dir, exist_ok=True)
    os.makedirs(final_dir, exist_ok=True)
    os.makedirs(camera_dir, exist_ok=True)

    extrinsic_path = args.input_dir + "/cameras/camera_extrinsics.json" 
    with open(extrinsic_path, "r") as f:
        all_extrinsics = json.load(f)

    # cam01 ~ cam10
    cam_ids = [f"cam{str(i).zfill(2)}.mp4" for i in range(1, 11)]
    # breakpoint()
    def read_video_frames(video_path):
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        return frames, fps

    def write_video_frames(frames, output_path, fps):
        h, w = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # temporary save
        out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
        for frame in frames:
            out.write(frame)
        out.release()

    def reencode_with_ffmpeg(input_path, output_path):
        cmd = [
            "ffmpeg", "-y", "-threads", "1", "-i", input_path,
            "-vcodec", "libx264", "-crf", "23", output_path
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    video_mapping = {}
    final_camera_extrinsics = {} 
    # main loop
    start_idx = random.randint(0, 80)
    
    for base_cam in cam_ids:
        base_path = os.path.join(input_dir, base_cam)

        # reverse playback of base cam video
        base_frames, base_fps = read_video_frames(base_path)
        base_frames_reversed = base_frames[::-1]

        # randomly select other cam
        candidate_cams = [cam for cam in cam_ids if cam != base_cam]
        selected_cam = random.choice(candidate_cams)
        selected_path = os.path.join(input_dir, selected_cam)
        selected_frames, _ = read_video_frames(selected_path)
        selected_frames = selected_frames[1:]  # exclude first frame

        # connect
        combined_frames = base_frames_reversed + selected_frames  # total 161 frames
        assert len(combined_frames) == 161

        # random start point selection (0~80)
  
        selected_clip = combined_frames[start_idx:start_idx + 81]
        assert len(selected_clip) == 81

        # path setup
        tmp_output_path = os.path.join(tmp_dir, base_cam)
        final_output_path = os.path.join(final_dir, base_cam)

        # save
        write_video_frames(selected_clip, tmp_output_path, base_fps)
        reencode_with_ffmpeg(tmp_output_path, final_output_path)

        video_mapping[base_cam[:-4]] = {
        "other_cam": selected_cam[:-4],
        "start_idx": start_idx,
        "fps": base_fps
        }

        print(f"{base_cam} saved successfully (attached cam: {selected_cam}, fps: {base_fps}, start frame: {start_idx})")

    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)

    final_camera_extrinsics = {}

    frame_keys = sorted(all_extrinsics.keys(), key=lambda x: int(x.replace("frame", "")))
    ref_data = {}
    N = len(frame_keys)
    for i, old_key in enumerate(frame_keys):
        new_key = f"frame{N - 1 - i}"
        ref_data[new_key] = all_extrinsics[old_key]

    for i, old_key in enumerate(frame_keys):
        if i==0:
            continue
        new_key = f"frame{N - 1 + i}"
        ref_data[new_key] = all_extrinsics[old_key]

    final_camera_extrinsics = {}
    cam_ids = [f"cam{str(i).zfill(2)}.mp4" for i in range(1, 11)]

    for frame_idx in range(81):
        frame_key = f"frame{frame_idx}"
        final_camera_extrinsics[frame_key] = {}

        for cam_id in cam_ids:
            cam_name = cam_id[:-4]

            mapping = video_mapping[cam_name]
            base_cam = cam_name
            other_cam = mapping["other_cam"]
            start_idx = mapping["start_idx"]

            # 81 frames = [reversed base (start_idx:80)] + [other (1:)]
            point = start_idx + frame_idx 
            split_point = 81

            if point < split_point:
                # reversed base_cam frame
                src_frame = f"frame{point}"
                src_cam = base_cam
            else:
                src_frame = f"frame{point}"
                src_cam = other_cam

            try:
                final_camera_extrinsics[frame_key][cam_name] = ref_data[src_frame][src_cam]
                #print(f"frame{frame_idx} saved (current cam: {src_cam}, start frame: {src_frame}, current point: {point})")
            except:
                print(f"[WARNING] Missing {src_cam} in {src_frame}")

    output_path = os.path.join(camera_dir, "video_mapping.json")
    with open(output_path, "w") as f:
        json.dump(video_mapping, f, indent=4)

    extrinsic_output_path = os.path.join(camera_dir, "camera_extrinsics.json")
    with open(extrinsic_output_path, "w") as f:
        json.dump(final_camera_extrinsics, f, indent=4)

    # extrinsic_output_path_ori = os.path.join(camera_dir, "camera_extrinsics_ori.json")
    # with open(extrinsic_output_path_ori, "w") as f:
    #     json.dump(all_extrinsics, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True, help="Path to input video directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to temporary output directory")

    args = parser.parse_args()
    main(args)