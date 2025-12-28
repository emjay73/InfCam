import os
import subprocess
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import argparse

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

def run(cmd):
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["VECLIB_MAXIMUM_THREADS"] = "1"
    env["NUMEXPR_NUM_THREADS"] = "1"

    #print(f"[RUNNING] {' '.join(cmd)}")
    # subprocess.run(cmd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    result = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"\n[ERROR] Command failed: {' '.join(cmd)}")
        print(f"[STDERR] {result.stderr}")
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_mcv", type=str, default="DATA/MultiCamVideo-Dataset/train",
                        help="Path to the MultiCamVideo-Dataset training directory")
    parser.add_argument("--path_augmcv", type=str, default=None,
                        help="Path to the Augmented MultiCamVideo-Dataset directory")                    
    args = parser.parse_args()

    num_workers = min(cpu_count(), 16)
    PATH_MCV = args.path_mcv

    if args.path_augmcv is None:
        from datetime import datetime
        date_str = datetime.now().strftime('%Y%m%d')
        PATH_AUG_MCV = f"./DATA/AugMCV_{date_str}"
    else:
        PATH_AUG_MCV = args.path_augmcv

    SCRIPT_PATH = "./preproc/aug_trajectory.py"

    # Prepare task list for parallel processing
    tasks = []
    for setting_name in os.listdir(PATH_MCV):
        setting_path = os.path.join(PATH_MCV, setting_name)
        if not os.path.isdir(setting_path):
            continue
        for scene_name in os.listdir(setting_path):

            # debug
            # if not 3385<int(scene_name.replace('scene', ''))<3395:
            #     continue

            for i in range(2):  # 0~4
                input_dir = os.path.join(PATH_MCV, setting_name, scene_name)
                output_dir = os.path.join(PATH_AUG_MCV, setting_name, f"{scene_name}_{i}")
                cmd = [
                    "python", SCRIPT_PATH,
                    "--input_dir", input_dir,
                    "--output_dir", output_dir
                ]
                tasks.append(cmd)
                
    with Pool(num_workers) as pool:
        print(f"Processing {len(tasks)} tasks with {num_workers} workers")
        print("It may take a while to initialize the workers...")
        for _ in tqdm(pool.imap_unordered(run, tasks), total=len(tasks)):
            pass
        