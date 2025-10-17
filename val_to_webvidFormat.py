import argparse, os, json, shutil
from os.path import join as opj
from glob import glob
import pandas as pd
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument("--video_dir", type=str, required=True)
parser.add_argument("--is_gt", action="store_true")
args = parser.parse_args()

n_save = n_skip = 0
if args.is_gt:
    metadata = pd.read_csv(opj(args.video_dir, "metadata_static_video.csv"))

    save_video_dir = args.video_dir.replace("val", "val_webvidFormat")
    new_matedata = {
        "file_name": [],
        "text": []
    }
    
    for i, (fn, txt) in enumerate(zip(metadata["file_name"], metadata["text"])):
        video_p = opj(args.video_dir, fn)
        save_video_p = opj(save_video_dir, f"videos/video{i}.mp4")
        save_cam_ref0_p = opj(save_video_dir, f"cameras/ref0/camera_extrinsics_{i}.json")

        new_matedata["file_name"].append(f"video{i}.mp4")
        new_matedata["text"].append(txt)

        cam_ref0_p = opj(os.path.dirname(video_p), "cameras/camera_extrinsic_refcam10.json")
        
        os.makedirs(os.path.dirname(save_video_p), exist_ok=True)
        os.makedirs(os.path.dirname(save_cam_ref0_p), exist_ok=True)
        if not os.path.exists(save_video_p):
            shutil.copy(video_p, save_video_p)
            n_save += 1
        else:
            n_skip += 1
        if not os.path.exists(save_cam_ref0_p):
            shutil.copy(cam_ref0_p, save_cam_ref0_p)
    save_metadata_p = opj(save_video_dir, "metadata.csv")
    if not os.path.exists(save_metadata_p):
        pd.DataFrame(new_matedata).to_csv(opj(save_video_dir, "metadata.csv"), index=False)
else:
    cnt = defaultdict(int)  # cam_type → 저장 개수(0..MAX_PER_CAM-1)
    f10_orig = sorted(glob(opj(args.video_dir, "f18_aperture10/*/videos/cam*.mp4")))
    f10_aug  = sorted(glob(opj(args.video_dir, "f18_aperture10_aug/*/videos/cam*.mp4")))
    f10_ps = [p for pair in zip(f10_orig, f10_aug) for p in pair]

    f24_orig = sorted(glob(opj(args.video_dir, "f24_aperture5/*/videos/cam*.mp4")))
    f24_aug  = sorted(glob(opj(args.video_dir, "f24_aperture5_aug/*/videos/cam*.mp4")))
    f24_ps = [p for pair in zip(f24_orig, f24_aug) for p in pair]

    f35_orig = sorted(glob(opj(args.video_dir, "f35_aperture2.4/*/videos/cam*.mp4")))
    f35_aug  = sorted(glob(opj(args.video_dir, "f35_aperture2.4_aug/*/videos/cam*.mp4")))
    f35_ps = [p for pair in zip(f35_orig, f35_aug) for p in pair]

    f50_ps = sorted(glob(opj(args.video_dir, "f50_aperture2.4/*/videos/cam*.mp4")))
    video_ps = f10_ps + f24_ps + f35_ps + f50_ps
    
    for p in video_ps:
        cam_type = int(os.path.basename(p).split(".")[0][3:])
        save_video_dir = opj(args.video_dir, "webvidFormat", f"cam_type{cam_type}")
        os.makedirs(save_video_dir, exist_ok=True)

        idx = cnt[cam_type]
        save_video_p = opj(save_video_dir, f"video{idx}.mp4")
        if not os.path.exists(save_video_p):
            shutil.copy(p, save_video_p)   # 덮어쓰기
            cnt[cam_type] += 1
            n_save += 1
        else:
            n_skip += 1
print(f"Saved {n_save} videos, skipped {n_skip} existing ones.")
    

