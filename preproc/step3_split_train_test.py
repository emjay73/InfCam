import os
import shutil
import argparse
import random
import cv2
import numpy as np
import imageio
import json
import pandas as pd
from tqdm import tqdm

list_test_scenes = [
    "f18_aperture10/scene3389",     
    "f18_aperture10/scene3390",     
    "f18_aperture10/scene3391",      
    "f18_aperture10/scene3392",                                                                                                                                              
    "f18_aperture10/scene3393",                                                                                                                                              
    "f18_aperture10/scene3394",                                                                                                                                              
    "f18_aperture10/scene3395",                                                                                                                                              
    "f18_aperture10/scene3396",                                                                                                                                              
    "f18_aperture10/scene3397",                                                                                                                                              
    "f18_aperture10/scene3398",                                                                                                                                              
    "f18_aperture10/scene3399",                                                                                                                                              
    "f18_aperture10/scene3400",
    "f18_aperture10_aug/scene3389",                                                                                                                                          
    "f18_aperture10_aug/scene3390",                                                                                                                                          
    "f18_aperture10_aug/scene3391",                                                                                                                                          
    "f18_aperture10_aug/scene3392",                                                                                                                                          
    "f18_aperture10_aug/scene3393",                                                                                                                                          
    "f18_aperture10_aug/scene3394",
    "f18_aperture10_aug/scene3395",
    "f18_aperture10_aug/scene3396",
    "f18_aperture10_aug/scene3397",
    "f18_aperture10_aug/scene3398",
    "f18_aperture10_aug/scene3399",
    "f18_aperture10_aug/scene3400",
    "f24_aperture5/scene3389",
    "f24_aperture5/scene3390",
    "f24_aperture5/scene3391",
    "f24_aperture5/scene3392",
    "f24_aperture5/scene3393",
    "f24_aperture5/scene3394",
    "f24_aperture5/scene3395",
    "f24_aperture5/scene3396",
    "f24_aperture5/scene3397",
    "f24_aperture5/scene3398",
    "f24_aperture5/scene3399",
    "f24_aperture5/scene3400",
    "f24_aperture5_aug/scene3389",
    "f24_aperture5_aug/scene3390",
    "f24_aperture5_aug/scene3391",
    "f24_aperture5_aug/scene3392",
    "f24_aperture5_aug/scene3393",
    "f24_aperture5_aug/scene3394",
    "f24_aperture5_aug/scene3395",
    "f24_aperture5_aug/scene3396",
    "f24_aperture5_aug/scene3397",
    "f24_aperture5_aug/scene3398",
    "f24_aperture5_aug/scene3399",
    "f24_aperture5_aug/scene3400",
    "f35_aperture2.4/scene3389",                                                                                                                                             "f35_aperture2.4/scene3390",                                                                                                                                             "f35_aperture2.4/scene3391",                                                                                                                                             "f35_aperture2.4/scene3392",                                                                                                                                             "f35_aperture2.4/scene3393",                                                                                                                                             "f35_aperture2.4/scene3394",                                                                                                                                             "f35_aperture2.4/scene3395",                                                                                                                                             
    "f35_aperture2.4/scene3396",
    "f35_aperture2.4/scene3397",
    "f35_aperture2.4/scene3398",
    "f35_aperture2.4/scene3399",
    "f35_aperture2.4/scene3400",
    "f35_aperture2.4_aug/scene3389",
    "f35_aperture2.4_aug/scene3390",
    "f35_aperture2.4_aug/scene3391",
    "f35_aperture2.4_aug/scene3392",
    "f35_aperture2.4_aug/scene3393",
    "f35_aperture2.4_aug/scene3394",
    "f35_aperture2.4_aug/scene3395",
    "f35_aperture2.4_aug/scene3396",
    "f35_aperture2.4_aug/scene3397",
    "f35_aperture2.4_aug/scene3398",
    "f35_aperture2.4_aug/scene3399",
    "f35_aperture2.4_aug/scene3400",
]


def move_scene(scene_rel_path, dst_root):
    # scene_rel_path example: f18_aperture10_aug/scene3395
    src_path = os.path.join(path_augmcv, scene_rel_path)
    parent, scene_name = os.path.split(scene_rel_path)
    dst_parent = os.path.join(dst_root, parent)
    os.makedirs(dst_parent, exist_ok=True)
    dst_path = os.path.join(dst_parent, scene_name)
    if os.path.exists(src_path):
        if os.path.exists(dst_path):
            print(f"Destination already exists for {dst_path}, skipping move.")
        else:
            shutil.move(src_path, dst_path)
            if 'test' in dst_path:
                print(f"Moved {src_path} -> {dst_path}")
    else:
        print(f"Source {src_path} does not exist, skipping.")

def move_test_scenes_to_test_folder(path_augmcv, list_test_scenes, test_dir):
    # move scenes in list_test_scenes to test folder
    for scene_rel in list_test_scenes:
        parent, scene_name = os.path.split(scene_rel)
        parent_full_path = os.path.join(path_augmcv, parent)
        if not os.path.isdir(parent_full_path):
            print(f"Parent directory {parent_full_path} does not exist, skipping.")
            continue
        # move all folders starting with scene_name in the parent folder to test folder
        for item in os.listdir(parent_full_path):
            if item.startswith(scene_name):
                rel_path = os.path.join(parent, item)
                move_scene(rel_path, test_dir)

def move_remaining_scenes_to_train(path_augmcv, list_test_scenes, train_dir):
    # move remaining scene folders (not moved in the previous step) to train folder
    for setting_name in os.listdir(path_augmcv):
        setting_path = os.path.join(path_augmcv, setting_name)
        # skip train and test folders
        if setting_name in ["train", "test"]:
            continue
        if not os.path.isdir(setting_path):
            continue
        for scene_name in os.listdir(setting_path):
            scene_path = os.path.join(setting_path, scene_name)
            if not os.path.isdir(scene_path):
                continue
            scene_rel_path = os.path.join(setting_name, scene_name)
            # skip if already moved to test folder
            if scene_rel_path in list_test_scenes:
                continue
            move_scene(scene_rel_path, train_dir)
    
    remove_empty_non_train_test_dirs(path_augmcv)


def remove_empty_non_train_test_dirs(path_root):
    """
    Remove all empty directories in path_root except 'train' and 'test' directories.
    If a directory is not empty, raises a RuntimeError.
    """
    for item in os.listdir(path_root):
        item_path = os.path.join(path_root, item)
        if item not in ["train", "test"] and os.path.isdir(item_path):
            if not os.listdir(item_path):  # folder is empty
                try:
                    os.rmdir(item_path)
                    print(f"Removed empty leftover directory: {item_path}")
                except Exception as e:
                    print(f"Failed to remove {item_path}: {e}")
            else:
                raise RuntimeError(f"Directory {item_path} is not empty. Remove contents before deleting.")

def process_src_videos(path_video_in, path_video_out, start_idx, n_frames):
    
    cap = cv2.VideoCapture(path_video_in)
    
    if not cap.isOpened():
        print(f"Error opening video file {path_video_in}")
        return

    # Get original dimensions
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = imageio.get_writer(path_video_out, fps=15, quality=9, ffmpeg_params=None)
    # print(f"Processing {output_video_path}...")
    
    # Process each frame
    list_frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Write the frame to the output video
        list_frames.append(frame_rgb)

    for i, frame in enumerate(list_frames[::-1] + list_frames):   
        if i < start_idx or i >= start_idx + n_frames:
            continue
        out.append_data(frame)

    # Release everything if job is finished
    cap.release()
    out.close()

    print(f"Processed {path_video_out}")

def path_video_out_existance_check(path_video_out):

    exists = False
    if os.path.exists(path_video_out):
        try:
            with imageio.get_reader(path_video_out) as reader:
                reader.get_meta_data()
            print(f"{path_video_out} already exists and is valid")
            exists = True
        except Exception as e:
            print(f"File {path_video_out} exists but is invalid, reprocessing: {e}")
            exists = False

    return exists

def copy_static_video_from_mcv_to_augmcv(path_augmcv, path_mcv):
    """
    Copies static videos from MultiCamVideo (mcv) reference set to augmcv test set,
    creating static_video.mp4 for each scene in the augmcv test folder.
    Additionally, collects corresponding lines from path_mcv/metadata.csv for these source videos
    and saves them as metadata_static_video.csv under the augmcv test path.
    """
    import pandas as pd

    path_phase = os.path.join(path_augmcv, 'test')
    # path_phase_ref = os.path.join(path_mcv, 'train')  # path_mcv is not split into train and test, just train
    path_phase_ref = path_mcv

    # Load metadata.csv from path_mcv for later reference
    # path_metadata_mcv = os.path.join(path_mcv, "metadata.csv")
    path_metadata_mcv = "preproc/metadata_mcv.csv"
    if not os.path.isfile(path_metadata_mcv):
        raise FileNotFoundError(f"{path_metadata_mcv} does not exist!")
    mcv_metadata = pd.read_csv(path_metadata_mcv)
    static_metadata_rows = []

    for subdir in sorted(os.listdir(path_phase)):
        path_subdir = os.path.join(path_phase, subdir)
        path_subdir_ref = os.path.join(path_phase_ref, subdir.replace("_aug", ""))

        if not os.path.isdir(path_subdir):
            print(f"Skipping {path_subdir} because it is not a directory")
            continue

        for idx, scene in enumerate(sorted(os.listdir(path_subdir))):
            path_scene = os.path.join(path_subdir, scene)
            # Use only the number part before '_' for reference: e.g., scene1234_x -> scene1234
            scene_num = scene.split("_")[0]
            path_scene_ref = os.path.join(path_subdir_ref, scene_num)

            path_video = os.path.join(path_scene_ref, "videos/cam10.mp4")
            path_video_out = os.path.join(path_scene, "static_video.mp4")

            vid_exists = path_video_out_existance_check(path_video_out)

            if not vid_exists:
                path_video_mapping = os.path.join(path_scene, "cameras/video_mapping.json")
                with open(path_video_mapping, 'r') as f:
                    video_mapping = json.load(f)

                start_idx = video_mapping["cam10"]["start_idx"]
                process_src_videos(path_video, path_video_out, start_idx, n_frames=81)

            # Find the row from mcv_metadata whose file_name matches the reference clip for cam10.mp4
            # Assume in mcv_metadata, file_name uses slashes and contains .../{scene}/videos/cam10.mp4
            # Find the relative path for matching (from path_mcv root)
            rel_scene_ref = os.path.relpath(path_scene_ref, path_phase_ref)
            candidate_file = os.path.join(rel_scene_ref, "videos/cam10.mp4")
            candidate_file = candidate_file.replace("\\", "/")  # Normalize for Windows/POSIX

            row_hit = mcv_metadata[mcv_metadata["file_name"] == candidate_file]
            if not row_hit.empty:
                row_dict = row_hit.iloc[0].to_dict()
                # Use replace to swap cam10.mp4 with static_video.mp4 in file_name
                row_dict["file_name"] = row_dict["file_name"].replace("cam10.mp4", "static_video.mp4")
                static_metadata_rows.append(row_dict)
            else:
                print(f"Warning: Could not find metadata entry for \"{candidate_file}\"")

    # After collecting all relevant metadata rows, write them to metadata_static_video.csv under path_phase
    if static_metadata_rows:
        meta_df = pd.DataFrame(static_metadata_rows)
        out_meta_path = os.path.join(path_phase, "metadata_static_video.csv")
        meta_df.to_csv(out_meta_path, index=False)
        print(f"Saved static video metadata to {out_meta_path}")
    else:
        print("No matching metadata entries found. metadata_static_video.csv NOT saved.")


def update_metadata_with_focallength_aug(path_augmcv):
    """
    Update metadata files for train and test splits by including augmented data entries.
    """
    for phase in ['train', 'test']:
        path_meta_src = os.path.join("preproc/metadata_augmcv_traj_" + phase + ".csv")
        path_meta_trg = os.path.join(path_augmcv, "metadata_augmcv_" + phase + ".csv")

        metadata = pd.read_csv(path_meta_src)
        file_list = metadata["file_name"].to_list()
        text_list = metadata["text"].to_list()

        path_phase = os.path.join(path_augmcv, phase)

        new_entries = []
        for file_name, text in tqdm(zip(file_list, text_list)):

            new_entries.append({'file_name': file_name, 'text': text}) # updated. should be here to include f50

            # Modify the file_name to include '_aug' in the first folder
            parts = file_name.split('/')
            parts[0] += '_aug'

            # Find the corresponding _aug folder
            aug_dir = os.path.join(path_phase, parts[0])

            if not os.path.exists(aug_dir):
                print(f"No matching folder found for {file_name}")
                continue

            scene_prefix = parts[1]
            matching_folders = [d for d in os.listdir(aug_dir) if d.startswith(scene_prefix)]

            if matching_folders:
                # Replace the scene number with the matching folder name
                parts[1] = matching_folders[0]
            else:
                print(f"No matching folder found for {file_name}")
                continue

            new_file_name = '/'.join(parts)

            # Add the new entry with the modified file_name
            # new_entries.append({'file_name': file_name, 'text': text})
            new_entries.append({'file_name': new_file_name, 'text': text})

        # Create a new DataFrame from the new_entries
        new_metadata = pd.DataFrame(new_entries)

        # Save the new metadata to the target CSV file
        print(f"Saving {path_meta_trg}...")
        new_metadata.to_csv(path_meta_trg, index=False)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Split dataset scenes into train and test directories.")
    parser.add_argument("--path_augmcv", type=str, default="DATA/AugMCV_20251226", help="Path to the dataset root directory")
    parser.add_argument("--path_mcv", type=str, default="DATA/MultiCamVideo-Dataset", help="Path to the dataset root directory")
    args = parser.parse_args()

    path_augmcv = args.path_augmcv
    path_mcv = args.path_mcv

    # Define train and test folders
    train_dir = os.path.join(path_augmcv, "train")
    test_dir = os.path.join(path_augmcv, "test")

    # Create train and test directories if they do not exist
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Call the function
    move_test_scenes_to_test_folder(path_augmcv, list_test_scenes, test_dir)
    move_remaining_scenes_to_train(path_augmcv, list_test_scenes, train_dir)    
    print("Train/test split and move completed.")

    copy_static_video_from_mcv_to_augmcv(path_augmcv, path_mcv)
    print("Static video copy completed.")
    
    update_metadata_with_focallength_aug(path_augmcv)
    print("Update metadata with augmented data completed.")


            
            

