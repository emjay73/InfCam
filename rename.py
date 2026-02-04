import os
import argparse
from pathlib import Path

def is_video_file(filename):
    """Check if file is a video file based on extension"""
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv', '.m4v', '.3gp', '.ogv'}
    return Path(filename).suffix.lower() in video_extensions

def rename_videos_in_folder(folder_path):
    """Rename videos in a folder to video1, video2, etc. in sorted order"""
    # Find all video files in the folder
    video_files = []
    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)
        if os.path.isfile(file_path) and is_video_file(file):
            video_files.append(file)
    
    if len(video_files) == 0:
        return False  # No videos in this folder
    
    # Sort video files by name
    video_files.sort()
    
    # Rename videos
    renamed_count = 0
    for idx, old_name in enumerate(video_files, start=1):
        old_path = os.path.join(folder_path, old_name)
        # Get original extension
        ext = Path(old_name).suffix
        new_name = f"video{idx}{ext}"
        new_path = os.path.join(folder_path, new_name)
        
        # Skip if already renamed
        if old_name == new_name:
            continue
        
        # Check if target name already exists and is different file
        if os.path.exists(new_path) and os.path.abspath(old_path) != os.path.abspath(new_path):
            print(f"Warning: {new_path} already exists. Skipping {old_name}")
            continue
        
        try:
            os.rename(old_path, new_path)
            print(f"Renamed: {old_name} -> {new_name}")
            renamed_count += 1
        except Exception as e:
            print(f"Error renaming {old_name}: {e}")
    
    if renamed_count > 0:
        print(f"Renamed {renamed_count} video(s) in {folder_path}\n")
        return True
    return False

def process_directory(root_path):
    """Recursively process all subdirectories"""
    root_path = os.path.abspath(root_path)
    
    if not os.path.exists(root_path):
        print(f"Error: Path does not exist: {root_path}")
        return
    
    if not os.path.isdir(root_path):
        print(f"Error: Path is not a directory: {root_path}")
        return
    
    print(f"Processing directory: {root_path}\n")
    
    # Walk through all subdirectories
    processed_folders = 0
    for root, dirs, files in os.walk(root_path):
        # Check if current folder has video files
        if rename_videos_in_folder(root):
            processed_folders += 1
    
    print(f"\nProcessed {processed_folders} folder(s) with videos")

def main():
    parser = argparse.ArgumentParser(description="Rename videos in subdirectories to video1, video2, etc.")
    parser.add_argument(
        "path",
        type=str,
        nargs='?',
        default=None,
        help="Root path to process (default: from path_input variable in script)"
    )
    
    args = parser.parse_args()
    
    # Get path from argument or from script variable
    if args.path:
        path_input = args.path
    else:
        # Try to read from script variable (for backward compatibility)
        # path_input = "/home/emjay_workspace/repo/custom/InfCam_extreme/results"
        path_input = "/home/emjay_workspace/repo/custom/InfCam_extreme/sample_data"
        print(f"Using default path: {path_input}")
    
    process_directory(path_input)

if __name__ == "__main__":
    main()
