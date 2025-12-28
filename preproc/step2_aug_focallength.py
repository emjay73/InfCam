import random
import os
import cv2
import numpy as np
import imageio
import multiprocessing
import argparse
from multiprocessing import cpu_count

f_list = [18, 24, 35, 50]

def process_videos(path_scene_in, path_scene_out, f_now, f_new):
    for i in range(1, 11):
        video_path = os.path.join(path_scene_in, f"videos/cam{i:02}.mp4")
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"Error opening video file {video_path}")
            continue

        # Get original dimensions
        original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        new_H = int(f_new / f_now * original_height)
        new_W = int(f_new / f_now * original_width)

        # Define output video path
        output_video_path = os.path.join(path_scene_out, f"videos/cam{i:02}.mp4")
        os.makedirs(os.path.dirname(output_video_path), exist_ok=True)

        # Define the codec and create VideoWriter object
        # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        # out = cv2.VideoWriter(output_video_path, fourcc, 30.0, (original_width, original_height))
        out = imageio.get_writer(output_video_path, fps=15, quality=9, ffmpeg_params=None)
        # print(f"Processing {output_video_path}...")
        
        # Process each frame
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Resize to new dimensions first
            resized_frame = cv2.resize(frame_rgb, (new_W, new_H), interpolation=cv2.INTER_CUBIC)

            # Center crop
            center_h = new_H // 2
            center_w = new_W // 2
            start_h = max(center_h - original_height // 2, 0)
            end_h = min(center_h + original_height // 2, new_H)
            start_w = max(center_w - original_width // 2, 0)
            end_w = min(center_w + original_width // 2, new_W)

            cropped_frame = resized_frame[start_h:end_h, start_w:end_w]

            # Calculate padding if needed
            pad_h = (original_height - (end_h - start_h)) // 2
            pad_w = (original_width - (end_w - start_w)) // 2

            # Apply zero padding
            padded_frame = cv2.copyMakeBorder(cropped_frame, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_CONSTANT, value=[0, 0, 0])

            # Write the frame to the output video
            out.append_data(padded_frame)

        # Release everything if job is finished
        cap.release()
        out.close()

def process_scene(scene_info):
    path_scene, path_scene_aug, f_now, f_new = scene_info
    
    # Check if a folder with the same scene name exists in the output path
    scene_name = os.path.basename(path_scene_aug).split('_f')[0]
    existing_folders = [d for d in os.listdir(os.path.dirname(path_scene_aug)) if scene_name in d]

    if existing_folders:
        print(f"Skip: {existing_folders}")
        return

    print(f"\t Generating {path_scene_aug}...")

    os.makedirs(path_scene_aug, exist_ok=True)
    os.makedirs(path_scene_aug+"/cameras", exist_ok=True)
    os.system(f"cp {path_scene}/cameras/camera_extrinsics.json {path_scene_aug}/cameras/camera_extrinsics.json")
    os.system(f"cp {path_scene}/cameras/video_mapping.json {path_scene_aug}/cameras/video_mapping.json")
    process_videos(path_scene, path_scene_aug, f_now, f_new)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_data", type=str, default="DATA/AugMCV_20251226", help="Path to the AugMCV-Dataset directory")
    args = parser.parse_args()

    # f18_aperture10, ...
    for subdir in sorted(os.listdir(args.path_data)):
        
        f_new = f_now = int(subdir.split("_")[0][1:])
        
        path_subdir = os.path.join(args.path_data, subdir)
        path_subdir_aug = os.path.join(args.path_data, subdir+"_aug")
        os.makedirs(path_subdir_aug, exist_ok=True)
        print(f"Generating {path_subdir_aug}...")

        # scene1_0, ...
        # Create a shuffled list of focal lengths excluding f_now
        available_f_list = [f for f in f_list if f > f_now]
        
        random.shuffle(available_f_list)

        if len(available_f_list) == 0:
            print(f"No available focal lengths for {path_subdir}")
            continue
        
        # Ensure the list is long enough
        if len(available_f_list) < len(os.listdir(path_subdir)):
            available_f_list *= (len(os.listdir(path_subdir)) // len(available_f_list)) + 1
        
        
        scene_infos = []
        for idx, scene in enumerate(sorted(os.listdir(path_subdir))):
            f_new = available_f_list[idx]
            path_scene = os.path.join(path_subdir, scene)
            path_scene_aug = os.path.join(path_subdir_aug, scene+f"_f{f_new}")
            scene_infos.append((path_scene, path_scene_aug, f_now, f_new))

        # print("scene_infos", scene_infos)
        num_workers = min(cpu_count()//2, 16)
        with multiprocessing.Pool(processes=num_workers) as pool: #lscpu | grep "CPU(s):" | awk '{print $2}' 의 절반
            pool.map(process_scene, scene_infos)
