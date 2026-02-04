import os
import re
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
import argparse
import torch
import imageio
from PIL import Image
import torchvision
from torchvision.transforms import v2
from einops import rearrange
from unidepth.models import UniDepthV2

class Camera(object):
    """Camera class same as inference_infcam.py"""
    def __init__(self, c2w):
        c2w_mat = np.array(c2w).reshape(4, 4)
        self.c2w_mat = c2w_mat
        self.w2c_mat = np.linalg.inv(c2w_mat)

def parse_matrix(matrix_str):
    """Parse matrix string from JSON format"""
    rows = matrix_str.strip().split('] [')
    matrix = []
    for row in rows:
        row = row.replace('[', '').replace(']', '')
        matrix.append(list(map(float, row.split())))
    return np.array(matrix)

def get_relative_pose(cam_params):
    """Get relative pose same as InfCam_extreme/inference_infcam.py"""
    abs_w2cs = [cam_param.w2c_mat for cam_param in cam_params]
    abs_c2ws = [cam_param.c2w_mat for cam_param in cam_params]

    cam_to_origin = 0
    target_cam_c2w = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, -cam_to_origin],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    abs2rel = target_cam_c2w @ abs_w2cs[0]

    offset = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],                      
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ]) 
    ret_poses = [offset @ target_cam_c2w, ] + [offset @ abs2rel @ abs_c2w for abs_c2w in abs_c2ws[1:]]
 
    ret_poses = np.array(ret_poses, dtype=np.float32)
    return ret_poses

def crop_and_resize(image, target_width, target_height):
    """Crop and resize image same as inference_infcam.py"""
    width, height = image.size
    scale = max(target_width / width, target_height / height)
    image = torchvision.transforms.functional.resize(
        image,
        (round(height*scale), round(width*scale)),
        interpolation=torchvision.transforms.InterpolationMode.BILINEAR
    )
    return image

def load_video_frame(video_path, frame_id, height, width):
    """Load a single frame from video"""
    try:
        reader = imageio.get_reader(video_path)
        if frame_id >= reader.count_frames():
            reader.close()
            return None
        
        frame = reader.get_data(frame_id)
        frame = Image.fromarray(frame)
        frame = crop_and_resize(frame, width, height)
        
        frame_process = v2.Compose([
            v2.CenterCrop(size=(height, width)),
            v2.Resize(size=(height, width), antialias=True),
            v2.ToTensor(),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        frame = frame_process(frame)
        reader.close()
        return frame
    except Exception as e:
        print(f"Error loading frame from {video_path}: {e}")
        return None

def estimate_K_from_unidepth(video_path, height, width, device=None):
    """Estimate intrinsic from unidepth same as inference_infcam.py"""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    depth_type_ = "l"  # available types: s, b, l
    depth_name = f"unidepth-v2-vit{depth_type_}14"
    try:
        depth_model = UniDepthV2.from_pretrained("models/unidepth-v2-vitl14")
    except:
        # Fallback to huggingface
        depth_model = UniDepthV2.from_pretrained(f"lpiccinelli/{depth_name}")
    
    depth_model.interpolation_mode = "bilinear"
    depth_model = depth_model.to(device=device)

    try:
        # Load first few frames from video
        frames = []
        for i in range(min(5, 10)):  # Load up to 5 frames
            frame = load_video_frame(video_path, i, height, width)
            if frame is not None:
                frames.append(frame)
        
        if len(frames) == 0:
            return None
        
        # Stack frames: [T, C, H, W] -> [C, T, H, W]
        source_video = torch.stack(frames, dim=0)  # [T, C, H, W]
        source_video = rearrange(source_video, "T C H W -> C T H W")  # [C, T, H, W]
        source_video = source_video.unsqueeze(0)  # [1, C, T, H, W]
        
        # Permute for unidepth: [1, C, T, H, W] -> [1, T, C, H, W] -> [T, C, H, W]
        depth_input = source_video[0].permute(1, 0, 2, 3)  # [T, C, H, W]
        predictions = depth_model.infer(depth_input)
        K = predictions["intrinsics"].mean(dim=0)
        
        f_mean = (K[0,0] + K[1,1]) / 2
        focal_px = f_mean.item()
        focal_mm = None  # Cannot convert back to mm without sensor size
        
        return focal_px, focal_mm
    except Exception as e:
        print(f"Error in estimate_K_from_unidepth for {video_path}: {e}")
        return None
    finally:
        # Explicitly free GPU memory
        try:
            depth_model = depth_model.to("cpu")
            del depth_model
            if 'predictions' in locals():
                del predictions
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except:
            pass

def analyze_dataset(base_path, metadata_path, camera_extrinsics_path, num_frames=81, height=480, width=832, pose_interval=1, k_from_unidepth=False, output_dir=None):
    """Analyze camera pose distribution in the dataset"""
    
    # Load metadata
    metadata = pd.read_csv(metadata_path)
    paths = [os.path.join(base_path, "videos", file_name) for file_name in metadata["file_name"]]
    
    print(f"Found {len(paths)} videos in metadata")
    
    # Load camera extrinsics
    if not os.path.exists(camera_extrinsics_path):
        print(f"Error: Camera extrinsics file not found: {camera_extrinsics_path}")
        return
    
    with open(camera_extrinsics_path, 'r') as file:
        cam_data = json.load(file)
    
    # Storage for statistics
    rotation_angles = []
    translation_magnitudes = []
    focal_lengths_px = []
    focal_lengths_mm = []
    image_sizes = []
    principal_points = []
    
    # Process each video path
    for path in tqdm(paths, desc="Processing videos"):
        try:
            if not os.path.exists(path):
                continue
            
            # Calculate focal length
            if k_from_unidepth:
                # Use unidepth to estimate intrinsic
                result = estimate_K_from_unidepth(path, height, width)
                if result is not None:
                    focal_px, focal_mm = result
                    focal_lengths_px.append(focal_px)
                    focal_lengths_mm.append(focal_mm if focal_mm is not None else 0)
                else:
                    # Fallback to default if unidepth fails
                    focal_mm = 24
                    focal_px = focal_mm * max(height, width) / 23.76
                    focal_lengths_px.append(focal_px)
                    focal_lengths_mm.append(focal_mm)
            else:
                # Default focal length (same as inference_infcam.py)
                focal_mm = 24
                focal_px = focal_mm * max(height, width) / 23.76
                focal_lengths_px.append(focal_px)
                focal_lengths_mm.append(focal_mm)
            
            # Get video frame count (approximate - we'll use num_frames)
            # In inference, num_frames is used, and cam_idx is calculated based on pose_interval
            # cam_idx = list(range(num_frames*pose_interval))[::4*pose_interval]
            cam_idx = list(range(int(num_frames * pose_interval)))[::int(4 * pose_interval)]
            
            # Process each camera type (cam01, cam02, etc.)
            # Get all camera keys from first frame
            first_frame_key = f"frame{cam_idx[0]}"
            if first_frame_key not in cam_data:
                continue
            
            camera_keys = [key for key in cam_data[first_frame_key].keys() if key.startswith("cam")]
            
            for cam_key in camera_keys:
                # Load trajectory for this camera
                traj = []
                for frame_idx in cam_idx:
                    frame_key = f"frame{frame_idx}"
                    if frame_key not in cam_data:
                        continue
                    if cam_key not in cam_data[frame_key]:
                        continue
                    matrix_str = cam_data[frame_key][cam_key]
                    c2w = parse_matrix(matrix_str)
                    traj.append(c2w)
                
                if len(traj) == 0:
                    continue
                
                # Apply same transformations as inference_infcam.py
                traj = np.stack(traj).transpose(0, 2, 1)  # Transpose each matrix
                c2ws = []
                for c2w in traj:
                    c2w = c2w[:, [1, 2, 0, 3]]
                    c2w[:3, 1] *= -1.
                    c2w[:3, 3] /= 100
                    c2ws.append(c2w)
                
                # Create Camera objects
                cam_params = [Camera(cam_param) for cam_param in c2ws]
                
                # Calculate relative poses (first frame as reference)
                if len(cam_params) < 2:
                    continue
                
                # For each frame, calculate relative pose w.r.t. first frame
                for i in range(1, len(cam_params)):
                    relative_poses = get_relative_pose([cam_params[0], cam_params[i]])
                    relative_pose = relative_poses[1]  # Get the relative pose (second element)
                    
                    # Extract rotation and translation from relative pose
                    # relative_pose is 3x4 matrix (rotation 3x3 + translation 3x1)
                    rel_rot = relative_pose[:3, :3]
                    rel_trans = relative_pose[:3, 3]
                    
                    # Extract rotation angle
                    try:
                        rotation = R.from_matrix(rel_rot)
                        angle_rad = rotation.magnitude()
                        angle_deg = np.degrees(angle_rad)
                        rotation_angles.append(angle_deg)
                    except:
                        # Fallback
                        trace = np.trace(rel_rot)
                        angle_rad = np.arccos(np.clip((trace - 1) / 2, -1, 1))
                        angle_deg = np.degrees(angle_rad)
                        rotation_angles.append(angle_deg)
                    
                    # Extract translation magnitude (in meters, since /100 was already applied)
                    trans_mag = np.linalg.norm(rel_trans)
                    translation_magnitudes.append(trans_mag)
            
            # Store image size and principal point (should be constant)
            image_sizes.append((height, width))
            principal_points.append((width // 2, height // 2))
            
        except Exception as e:
            print(f"Error processing {path}: {e}")
            continue
    
    # Compute statistics
    output_lines = []
    output_lines.append("\n" + "="*80)
    output_lines.append("CAMERA POSE DISTRIBUTION ANALYSIS")
    output_lines.append("="*80)
    
    # Rotation angles
    if rotation_angles:
        rotation_angles = np.array(rotation_angles)
        output_lines.append(f"\n📐 ROTATION ANGLES (relative to first frame, degrees):")
        output_lines.append(f"   Min:     {np.min(rotation_angles):.6f}")
        output_lines.append(f"   Max:     {np.max(rotation_angles):.6f}")
        output_lines.append(f"   Mean:    {np.mean(rotation_angles):.6f}")
        output_lines.append(f"   Variance: {np.var(rotation_angles):.6f}")
        output_lines.append(f"   Std:     {np.std(rotation_angles):.6f}")
        output_lines.append(f"   Median:  {np.median(rotation_angles):.6f}")
        output_lines.append(f"   Count:   {len(rotation_angles)}")
    
    # Translation magnitudes
    if translation_magnitudes:
        translation_magnitudes = np.array(translation_magnitudes)
        output_lines.append(f"\n📍 TRANSLATION MAGNITUDES (relative to first frame, in meters after /100 scaling):")
        output_lines.append(f"   Min:     {np.min(translation_magnitudes):.6f}")
        output_lines.append(f"   Max:     {np.max(translation_magnitudes):.6f}")
        output_lines.append(f"   Mean:    {np.mean(translation_magnitudes):.6f}")
        output_lines.append(f"   Variance: {np.var(translation_magnitudes):.6f}")
        output_lines.append(f"   Std:     {np.std(translation_magnitudes):.6f}")
        output_lines.append(f"   Median:  {np.median(translation_magnitudes):.6f}")
        output_lines.append(f"   Count:   {len(translation_magnitudes)}")
    
    # Focal lengths
    if focal_lengths_px:
        focal_lengths_px = np.array(focal_lengths_px)
        focal_lengths_mm = np.array(focal_lengths_mm)
        output_lines.append(f"\n🔍 FOCAL LENGTHS (pixels):")
        output_lines.append(f"   Min:     {np.min(focal_lengths_px):.6f}")
        output_lines.append(f"   Max:     {np.max(focal_lengths_px):.6f}")
        output_lines.append(f"   Mean:    {np.mean(focal_lengths_px):.6f}")
        output_lines.append(f"   Variance: {np.var(focal_lengths_px):.6f}")
        output_lines.append(f"   Std:     {np.std(focal_lengths_px):.6f}")
        output_lines.append(f"   Median:  {np.median(focal_lengths_px):.6f}")
        output_lines.append(f"   Count:   {len(focal_lengths_px)}")
        
        output_lines.append(f"\n🔍 FOCAL LENGTHS (mm):")
        # Filter out None or 0 values if using unidepth
        focal_lengths_mm_filtered = [f for f in focal_lengths_mm if f is not None and f > 0]
        if focal_lengths_mm_filtered:
            unique_focal_mm = np.unique(focal_lengths_mm_filtered)
            output_lines.append(f"   Unique values: {sorted(unique_focal_mm)}")
            for focal_mm in sorted(unique_focal_mm):
                count = np.sum(np.array(focal_lengths_mm) == focal_mm)
                output_lines.append(f"   f{focal_mm}: {count} videos")
        else:
            output_lines.append(f"   (Using unidepth - mm values not available)")
    
    # Image sizes
    if image_sizes:
        unique_sizes = set(image_sizes)
        output_lines.append(f"\n🖼️  IMAGE SIZES:")
        if len(unique_sizes) == 1:
            size = list(unique_sizes)[0]
            output_lines.append(f"   All images have the same size: {size[0]}x{size[1]}")
        else:
            output_lines.append(f"   Found {len(unique_sizes)} different sizes:")
            for size in sorted(unique_sizes):
                count = image_sizes.count(size)
                output_lines.append(f"   {size[0]}x{size[1]}: {count} videos")
    
    # Principal points
    if principal_points:
        unique_pp = set(principal_points)
        output_lines.append(f"\n🎯 PRINCIPAL POINTS:")
        if len(unique_pp) == 1:
            pp = list(unique_pp)[0]
            output_lines.append(f"   All images have the same principal point: ({pp[0]}, {pp[1]})")
        else:
            output_lines.append(f"   Found {len(unique_pp)} different principal points:")
            for pp in sorted(unique_pp):
                count = principal_points.count(pp)
                output_lines.append(f"   ({pp[0]}, {pp[1]}): {count} videos")
    
    output_lines.append("\n" + "="*80)
    
    # Print to console
    for line in output_lines:
        print(line)
    
    # Save to file
    if output_dir is not None:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate output filename from camera_extrinsics_path and pose_interval
        camera_extrinsics_filename = os.path.basename(camera_extrinsics_path)
        # Remove extension
        camera_extrinsics_name = os.path.splitext(camera_extrinsics_filename)[0]
        output_filename = f"{camera_extrinsics_name}_pose_interval_{pose_interval}.txt"
        output_path = os.path.join(output_dir, output_filename)
        
        # Write to file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(output_lines))
        
        print(f"\nResults saved to: {output_path}")
    
    return "\n".join(output_lines)

def main():
    parser = argparse.ArgumentParser(description="Analyze camera pose distribution in dataset")
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="/home/emjay_workspace/repo/custom/InfCam_extreme/sample_data",
        help="Base path of the dataset",
    )
    parser.add_argument(
        "--metadata_file_name",
        type=str,
        default="metadata.csv",
        help="Name of metadata CSV file",
    )
    parser.add_argument(
        "--camera_extrinsics_path",
        type=str,
        default=None,
        help="Path to camera extrinsics JSON file",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=81,
        help="Number of frames per video",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Image height",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=832,
        help="Image width",
    )
    parser.add_argument(
        "--pose_interval",
        type=float,
        default=1,
        help="Pose interval (same as inference_infcam.py)",
    )
    parser.add_argument(
        "--k_from_unidepth",
        action="store_true",
        default=False,
        help="Use unidepth to estimate intrinsic parameters",
    )
    
    args = parser.parse_args()
    
    metadata_path = os.path.join(args.dataset_path, args.metadata_file_name)
    
    if not os.path.exists(metadata_path):
        print(f"Error: Metadata file not found: {metadata_path}")
        return
    
    # If camera_extrinsics_path is not provided, try to find one
    if args.camera_extrinsics_path is None:
        # Try common camera extrinsics files
        possible_paths = [
            os.path.join(args.dataset_path, "cameras", "camera_extrinsics_extended_ref0.json"),
            os.path.join(args.dataset_path, "cameras", "camera_extrinsics_ref0.json"),
            os.path.join(args.dataset_path, "cameras", "camera_extrinsics_10types.json"),
        ]
        for path in possible_paths:
            if os.path.exists(path):
                args.camera_extrinsics_path = path
                print(f"Using camera extrinsics: {path}")
                break
        
        if args.camera_extrinsics_path is None:
            print(f"Error: Camera extrinsics file not found. Please specify --camera_extrinsics_path")
            return
    elif not os.path.exists(args.camera_extrinsics_path):
        print(f"Error: Camera extrinsics file not found: {args.camera_extrinsics_path}")
        return
    
    # Create output directory
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "analysis")
    
    analyze_dataset(
        args.dataset_path,
        metadata_path,
        args.camera_extrinsics_path,
        num_frames=args.num_frames,
        height=args.height,
        width=args.width,
        pose_interval=args.pose_interval,
        k_from_unidepth=args.k_from_unidepth,
        output_dir=output_dir
    )

if __name__ == "__main__":
    main()

