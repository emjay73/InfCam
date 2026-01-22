def run_add_cam_traj():
    import json
    from ipdb import set_trace as breakpoint

    with open("./example_test_data/cameras/camera_extrinsics.json", 'r') as file: cam_data = json.load(file)
    with open("./example_test_data/cameras/cam11.txt") as f: cam11 = f.readlines()
    with open("./example_test_data/cameras/cam12.txt") as f: cam12 = f.readlines()
    with open("./example_test_data/cameras/cam13.txt") as f: cam13 = f.readlines()
    with open("./example_test_data/cameras/cam14.txt") as f: cam14 = f.readlines()
    with open("./example_test_data/cameras/cam15.txt") as f: cam15 = f.readlines()
    with open("./example_test_data/cameras/cam16.txt") as f: cam16 = f.readlines()
    with open("./example_test_data/cameras/cam17.txt") as f: cam17 = f.readlines()


    for name, cam in zip(["cam11", "cam12", "cam13", "cam14", "cam15", "cam16", "cam17"], [cam11, cam12, cam13, cam14, cam15, cam16, cam17]):
        for frame_idx, line in enumerate(cam):
            line = line.strip() + " "
            cam_data[f"frame{frame_idx}"][name] = line

    with open("./example_test_data/cameras/camera_extrinsics.json", 'w') as file:
        json.dump(cam_data, file, indent=2)
run_add_cam_traj()

def run_make_grid_single_folder():
    import os
    import imageio.v2 as imageio
    import numpy as np
    from PIL import Image

    def resize_frame(frame, w, h):
        return np.array(Image.fromarray(frame).resize((w, h), Image.BICUBIC))

    def concat_videos(input_dir, output_path, n_h, n_w, v_h, v_w):
        paths = sorted([
            os.path.join(input_dir, f)
            for f in os.listdir(input_dir)
            if f.endswith(".mp4")
        ])[:n_h * n_w]

        if len(paths) < n_h * n_w:
            raise ValueError(f"{n_h * n_w}개 비디오 필요, 현재 {len(paths)}개")

        readers = [imageio.get_reader(p) for p in paths]
        fps = min(r.get_meta_data()['fps'] for r in readers)
        frames = min(r.count_frames() for r in readers)
        writer = imageio.get_writer(output_path, fps=fps)

        for i in range(frames):
            imgs = [resize_frame(r.get_data(i), v_w, v_h) for r in readers]
            grid = np.vstack([
                np.hstack(imgs[r * n_w:(r + 1) * n_w])
                for r in range(n_h)
            ])
            writer.append_data(grid)

        writer.close()
        [r.close() for r in readers]

    concat_videos(
        "outputs/step2668/cam_type1",
        "outputs/step2668/cam_type1_grid.mp4",
        3, 2, 480, 832
    )
# run_make_grid_single_folder()

def run_make_grid_multi_folder():
    import os, cv2, numpy as np
    from imageio.v2 import get_writer
    def merge(folders, save_dir, n_row, n_col, h, w):
        os.makedirs(save_dir, exist_ok=True)
        names = sorted(set(os.listdir(folders[0])).intersection(*[set(os.listdir(f)) for f in folders[1:]]))
        for name in names:
            caps = [cv2.VideoCapture(os.path.join(f, name)) for f in folders]
            min_f = min([c.get(cv2.CAP_PROP_FRAME_COUNT) for c in caps])
            fps = min([c.get(cv2.CAP_PROP_FPS) for c in caps])
            black = np.zeros((h, w, 3), np.uint8)
            out_path = os.path.join(save_dir, name)
            vw = get_writer(out_path, fps=fps, codec='libx264', quality=8)
            for _ in range(int(min_f)):
                frames = [cv2.resize(f, (w, h)) if r else None for c in caps if (r:=c.read())[0] for f in [r[1]]] + [None]*(n_row*n_col - len(caps))
                if all(f is None for f in frames): break
                padded = [f if f is not None else black for f in frames]
                grid = cv2.vconcat([cv2.hconcat(padded[i*n_col:(i+1)*n_col]) for i in range(n_row)])
                vw.append_data(grid[:,:,::-1])
            [c.release() for c in caps]
            vw.close()
            print(f"Saved {out_path}")
    
    folder_lst = [
        f"./outputs/reimple_step20000/cam_type1",
        f"./output_videos"
    ]
    merge(folder_lst, f"dummy_asd", 2, 1, 320, 544*2)
# run_make_grid_multi_folder()

def run_make_metadata_ego4d():
    """
    1. camera있는것만 필터링
    2. camera가 2개 이상인 경우만 필터링
    """
    import pandas as pd
    from glob import glob
    from tqdm import tqdm
    import json
    import os

    csv = pd.read_csv("./DATA/Ego4D/metadata_all.csv")
    json_ps = sorted(glob("./DATA/Ego4D/annotations/ego_pose/camera_pose/*.json"))
    print(f"Found {len(json_ps)} json files.")

    pose_folder_names = []
    for p in json_ps:
        with open(p, "r") as f:
            jf = json.load(f)
        pose_folder_names.append(jf["metadata"]["take_name"])
    pose_folder_names.sort()

    new_file_name = []
    new_txt = []
    for fn, txt in tqdm(zip(csv["file_name"], csv["text"]), total=len(csv)):
        folder_name = fn.split("/")[-5]

        if folder_name not in pose_folder_names:
            continue
        
        with open(os.path.join("./DATA/Ego4D/annotations/ego_pose/camera_pose", f"{folder_name}.json"), "r") as f:
            jf = json.load(f)

        cam_keys = [k for k in jf.keys() if k.startswith("cam") or k.startswith("gp")]
        if len(cam_keys) < 2:
            continue

        new_file_name.append(fn)
        new_txt.append(txt)
    
    print(f"Found {len(new_file_name)} files with 2 or more cameras.")
    new_data = pd.DataFrame({"file_name": new_file_name, "text": new_txt})
    new_data.to_csv("./DATA/Ego4D/metadata.csv", index=False)
# run_make_metadata_ego4d()
    

def run_make_pth_metadata_ego4d():
    import pandas as pd
    from glob import glob
    import os
    from tqdm import tqdm
    import json

    csv = pd.read_csv("./DATA/Ego4D/metadata.csv")
    json_ps = sorted(glob("./DATA/Ego4D/annotations/ego_pose/camera_pose/*.json"))
    print(f"Found {len(json_ps)} json files.")

    pose_folder_names = []
    for p in json_ps:
        with open(p, "r") as f:
            jf = json.load(f)
        pose_folder_names.append(jf["metadata"]["take_name"])
    pose_folder_names.sort()

    file_name = []
    text = []

    for p, t in tqdm(zip(csv["file_name"], csv["text"]), total=len(csv)):
        folder_name = p.split("/")[-5]
        if folder_name not in pose_folder_names:
            continue

        cam_keys = []
        with open(os.path.join("./DATA/Ego4D/annotations/ego_pose/camera_pose", f"{folder_name}.json"), "r") as f:
            jf = json.load(f)

            for k in jf.keys():
                if k.startswith("cam") or k.startswith("gp"):
                    cam_keys.append(k)
            
        n_files = []
        for cam_key in cam_keys:
            tensor_ps = sorted(glob(os.path.join("./DATA/Ego4D/takes/", folder_name, f"*/*/*/{cam_key}/*.pth")))
            n_files.append(len(tensor_ps))
    
        if len(set(n_files)) == 1:
            ps = sorted(glob(os.path.join("./DATA/Ego4D/takes/", os.path.splitext(p)[0], "*41x*.pth")))
            for pp in ps:
                for cam_key in cam_keys:
                    if cam_key in pp:
                        file_name.append("/".join(pp.split("/")[4:]))
                        text.append(t)
                        break
        else:
            print("differnet number of pth", p)

    print(f"Found {len(file_name)} files.")
    new_data = pd.DataFrame({"file_name": file_name, "text": text})
    new_data.to_csv("./DATA/Ego4D/metadata_stride96_lowresol.csv", index=False)
# run_make_pth_metadata_ego4d()