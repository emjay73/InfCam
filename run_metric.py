import os
import subprocess
import numpy as np
import json
# import pandas as pd # pandas 임포트 추가
import multiprocessing as mp
from functools import partial
from tqdm import tqdm
import pdb
#COLMAP_PATH = os.path.expanduser("~/colmap_install/bin/colmap")
#GLOMAP_PATH = os.path.expanduser("~/.local/bin/glomap")

# def run_colmap_pipeline(image_dir, project_dir):
#     db_path = os.path.join(project_dir, "database.db")
#     sparse_dir = os.path.join(project_dir, "sparse")
#     os.makedirs(sparse_dir, exist_ok=True)

#     print("---------------- 1. Feature extraction... ----------------")
#     subprocess.run([
#         "colmap", "feature_extractor",
#         "--database_path", db_path,
#         "--image_path", image_dir,
#         "--SiftExtraction.use_gpu", "1"
#     ], check=True)

#     print("---------------- 2. Feature matching... ----------------")
#     subprocess.run([
#         "colmap", "exhaustive_matcher", #"sequential_matcher", 
#         "--database_path", db_path,
#         "--SiftMatching.use_gpu", "1"
#     ], check=True)

#     print("---------------- 3. Sparse reconstruction... ----------------")
#     subprocess.run([
#         "glomap", "mapper",
#         "--database_path", db_path,
#         "--image_path", image_dir,
#         "--output_path", sparse_dir
#     ], check=True)

#     print("---------------- 4. Convert to TXT... ----------------")
#     subprocess.run([
#         "colmap", "model_converter",
#         "--input_path", os.path.join(sparse_dir, "0"),
#         "--output_path", os.path.join(sparse_dir, "0"),
#         "--output_type", "TXT"
#     ], check=True)

#     return os.path.join(sparse_dir, "0")

def qvec_to_rotmat(qvec):
    w, x, y, z = qvec
    return np.array([
        [1 - 2*y*y - 2*z*z,      2*x*y - 2*w*z,        2*x*z + 2*w*y],
        [2*x*y + 2*w*z,          1 - 2*x*x - 2*z*z,  2*y*z - 2*w*x],
        [2*x*z - 2*w*y,          2*y*z + 2*w*x,      1 - 2*x*x - 2*y*y]
    ])

def parse_gt_pose(matrix_str):
    rows = matrix_str.strip().split('] [')
    matrix = []
    for row in rows:
        row = row.replace('[', '').replace(']', '')
        matrix.append(list(map(float, row.split())))
    matrix = np.array(matrix)
    #breakpoint()

    traj = matrix.transpose(1, 0)

    c2w = traj[:, [1, 2, 0, 3]]
    c2w[:3, 1] *= -1.
    c2w[:3, 3] /= 100

    return c2w

def run_vipe_pipeline(image_dir, project_dir):
    # subprocess.run([
    #     "vipe", "infer",        
    #     "--image_dir", image_dir,
    #     "-o", project_dir,        
    # ], check=True)


    cmd = [
        "python", "run.py",
        "pipeline=default",
        "streams=frame_dir_stream",
        f"streams.base_path={image_dir}",
        "pipeline.post.depth_align_model=null"
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        print("Success!")
        print(result.stdout)
    else:
        print("Error occurred:")
        print(result.stderr)
    
    # # python scripts/vipe_to_colmap.py vipe_results/ --sequence dog_example
    # vipe_results_path = "vipe_results/"
    # sequence_name = "dog_example"

    # result = subprocess.run([
    #     "python", "scripts/vipe_to_colmap.py",
    #     vipe_results_path,
    #     "--sequence", sequence_name
    # ], capture_output=True, text=True)

    # print("Return code:", result.returncode)
    # print("STDOUT:", result.stdout)
    # print("STDERR:", result.stderr)

        
# def parse_gt_pose(pose_str):

#     rows = pose_str.strip().split("] [")
#     rows[0] = rows[0].lstrip("[")
#     rows[-1] = rows[-1].rstrip("]")

#     mat = [list(map(float, row.strip().split())) for row in rows]
#     T = np.eye(4)
#     for i in range(3):
#         T[i, :3] = mat[i][:3]
#     T[:3, 3] = mat[3][:3]

#     T[:3, 3] /= 100.0 # Convert to meters

#     return T

# emjay mofidied ------------------------------------------------
def convert_to_relative_poses(pose_dict, base_pose):
    # base_pose = pose_dict[ref_frame_id]
    base_inv = np.linalg.inv(base_pose)
    return {fid: base_inv @ pose for fid, pose in pose_dict.items()}

# original ------------------------------------------------
# def convert_to_relative_poses(pose_dict, ref_frame_id):
#     base_pose = pose_dict[ref_frame_id]
#     base_inv = np.linalg.inv(base_pose)
#     return {fid: base_inv @ pose for fid, pose in pose_dict.items()}
# ------------------------------------------------
def compute_errors(images_txt_path, gt_json_path, cam_type="cam01"):
    with open(gt_json_path, "r") as f:
        gt_data = json.load(f)

    # Load GT poses
    gt_poses = {
        fid: parse_gt_pose(gt_data[fid][cam_type])
        for fid in gt_data
        if cam_type in gt_data[fid]
    }

    #breakpoint()
    # Load COLMAP poses
    colmap_poses2 = {}
    with open(images_txt_path, "r") as f:
        lines = f.readlines()

    i = 0
    # emjay added -----------------------------------------------
    first_frame_pose = None
    # ---------------------------------------------------------
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("#") or len(line) == 0:
            i += 1
            continue

        elems = line.split()
        image_name = elems[9]

        # emjay modified -----------------------------------------------
        # frame 0 is the first frame from the src video, so we need to ignore it.
        # frame_id = int(image_name.split(".")[0])
        frame_id = int(image_name.replace("images/", "").replace("frame_", "").split(".")[0])
        if frame_id == 0:
            frame_id = "src_first_frame"
        else:
            frame_id = "frame" + str(frame_id - 1)

        # original ------------------------------------------------
        # frame_id = "frame" + str(int(image_name.split(".")[0]))
        # ---------------------------------------------------------

        qvec = list(map(float, elems[1:5]))
        tvec = list(map(float, elems[5:8])) # This tvec is -R.T @ C (C: camera center in world)
        R = qvec_to_rotmat(qvec)

        # COLMAP outputs are typically World-to-Camera (W_T_C) poses (qvec, tvec)
        # To get Camera-to-World (C_T_W) or camera's position in world coordinates:
        # C_T_W = (W_T_C)^-1 = [R_wc^T  -R_wc^T t_wc]
        # Here R = R_wc, t = t_wc (COLMAP's convention)
        cam_to_world = np.eye(4)
        cam_to_world[:3, :3] = R.T # R_wc^T
        cam_to_world[:3, 3] = -R.T @ np.array(tvec).flatten() # -R_wc^T t_wc

        # emjay modified -----------------------------------------------
        if frame_id == "src_first_frame":
            first_frame_pose = cam_to_world
        else:
            colmap_poses2[frame_id] = cam_to_world  
        # original ---------------------------------------------------------        
        # colmap_poses2[frame_id] = cam_to_world
        # ---------------------------------------------------------
        i += 2

    # emjay added -----------------------------------------------
    if first_frame_pose is None:
        raise ValueError("❌ First frame pose is not found.")
    # ---------------------------------------------------------

    from collections import OrderedDict

    colmap_poses = OrderedDict(
        sorted(colmap_poses2.items(), key=lambda x: int(x[0].replace("frame", "")))
    )

    gt_poses = {
        fid: parse_gt_pose(gt_data[fid][cam_type])
        for fid in gt_data
        if cam_type in gt_data[fid]
    }


    import re

    
    common_frames = sorted(
        [f for f in gt_poses if f in colmap_poses],
        key=lambda x: int(re.findall(r'\d+', x)[0])
    )
    # emjay commented out -----------------------------------------------
    # if len(common_frames) < 2:
    #     raise ValueError("❌ Not enough valid frames with both GT and COLMAP.")

    # f0, f1 = common_frames[:2]
    # ---------------------------------------------------------

    

    # emjay modified -----------------------------------------------
    # Make relative poses (cam-to-world) w.r.t. frame0
    colmap_poses_rel = convert_to_relative_poses(colmap_poses, first_frame_pose)
    gt_poses_rel = gt_poses
    # original ------------------------------------------------
    # # Make relative poses (cam-to-world) w.r.t. frame0
    # colmap_poses_rel = convert_to_relative_poses(colmap_poses, f0)
    # gt_poses_rel = convert_to_relative_poses(gt_poses, f0)
    # ------------------------------------------------

    # ---------- 새 스케일 정규화: pairwise 거리 기반 ----------
    eps = 1e-9
    
    # 인접한 프레임 쌍들의 거리 계산
    # scale 계산시 시작 프레임 위치의 영향 제거하기 위해 segment를 활용하여 계산.
    pairwise_scales = []
    for i in range(len(common_frames) - 1):
        frame_i = common_frames[i]
        frame_j = common_frames[i + 1]  # 인접한 다음 프레임
        
        # GT trajectory segment length
        gt_i = gt_poses_rel[frame_i][:3, 3]
        gt_j = gt_poses_rel[frame_j][:3, 3]
        gt_dist = np.linalg.norm(gt_j - gt_i)
        
        # Predicted trajectory segment length
        pred_i = colmap_poses_rel[frame_i][:3, 3]
        pred_j = colmap_poses_rel[frame_j][:3, 3]
        pred_dist = np.linalg.norm(pred_j - pred_i)
        
        # Scale 계산 (s_ij = ||t_j^(A) - t_{i}^(A)|| / ||t_j^(B) - t_{i}^(B)||)
        if pred_dist > eps:  # 0 나누기 방지
            scale_ij = gt_dist / pred_dist
            pairwise_scales.append(scale_ij)
    
    if len(pairwise_scales) == 0:
        print(f"⚠️ Warning: No valid pairwise scales found. Using scale=1.0.")
        scale = 1.0
    else:
        # Median scale 사용 (robust estimation)
        scale = float(np.median(pairwise_scales))
        print(f"📏 Adjacent frames pairwise distance-based scale normalization: "
                f"median scale={scale:.6f} from {len(pairwise_scales)} adjacent pairs")
    
    # Apply scale to predicted poses
    scaled_colmap_poses_rel = {k: v.copy() for k, v in colmap_poses_rel.items()}
    for pose_key in scaled_colmap_poses_rel:
        scaled_colmap_poses_rel[pose_key][:3, 3] *= scale
    # ---------------------------------------------------------


    ate_rot, ate_trans = compute_ate(gt_poses_rel, scaled_colmap_poses_rel, common_frames)


    # # 디버깅을 위해 초반 5개 프레임의 t_gt, t_pred 출력
    # for fid in common_frames[:5]:
    #     print(f"{fid}: t_gt = {gt_poses_rel[fid][:3, 3]}, t_pred = {scaled_colmap_poses_rel[fid][:3, 3]}")
    
    # print(f"📐 RotErrs (rad): {np.degrees(rot_errs[:5])} (deg)") # degree로 출력
    # print(f"📍 TransErrs (L2^2): {trans_errs[:5]}")

    # RPE (Relative Pose Error) 계산
    rpe_rot, rpe_trans = compute_rpe(gt_poses_rel, scaled_colmap_poses_rel, common_frames)
    
    print(f"📊 ATE - RotErr: {np.degrees(ate_rot):.4f} deg, TransErr: {ate_trans:.4f} m")
    print(f"📊 RPE - RotErr: {np.degrees(rpe_rot):.4f} deg, TransErr: {rpe_trans:.4f} m")

    return ate_rot, ate_trans, rpe_rot, rpe_trans

def compute_ate(gt_poses_rel, pred_poses_rel, common_frames):
    trans_errs = []
    rot_errs = []

    for fid in common_frames:
        R_gt = gt_poses_rel[fid][:3, :3]
        R_pred = pred_poses_rel[fid][:3, :3]

        t_gt = gt_poses_rel[fid][:3, 3]
        t_pred = pred_poses_rel[fid][:3, 3]

        # Error 계산 함수 사용
        rot_err = rotation_error(R_pred, R_gt)
        trans_err = translation_error(t_pred, t_gt)

        # TransErr가 1m^2를 초과하는 경우만 경고
        # if trans_err > 1.0:
        if trans_err > 20.0:
             print(f"⚠️ Large TransErr @ {fid}: {trans_err:.4f} m")
            #  continue

        rot_errs.append(rot_err)
        trans_errs.append(trans_err)
    print(f"Number of valid frames evaluated: {len(trans_errs)}")
    # return np.sum(rot_errs), np.sum(trans_errs)
    return np.mean(rot_errs), np.mean(trans_errs)

def rotation_error(R_pred, R_gt):
    """
    Rotation error 계산 함수
    """
    R_diff = R_pred @ R_gt.T
    trace = np.clip((np.trace(R_diff) - 1) / 2, -1.0, 1.0)
    rot_err = np.arccos(trace)  # in radians
    return rot_err

def translation_error(t_pred, t_gt):
    """
    Translation error 계산 함수 (ATE 방식과 동일)
    """
    trans_err = np.linalg.norm(t_pred - t_gt) 
    return trans_err

def compute_rpe(gt_poses_rel, pred_poses_rel, common_frames):
    """
    Relative Pose Error (RPE) 계산 함수
    """
    trans_errors = []
    rot_errors = []
    
    # common_frames를 정렬하여 순서대로 처리
    sorted_frames = sorted(common_frames)
    
    for i in range(len(sorted_frames)-1):
        frame1 = sorted_frames[i]
        frame2 = sorted_frames[i+1]
        
        # GT relative pose
        gt1 = gt_poses_rel[frame1]
        gt2 = gt_poses_rel[frame2]
        gt_rel = np.linalg.inv(gt1) @ gt2

        # Predicted relative pose
        pred1 = pred_poses_rel[frame1]
        pred2 = pred_poses_rel[frame2]
        pred_rel = np.linalg.inv(pred1) @ pred2
        
        # Relative error
        rel_err = np.linalg.inv(gt_rel) @ pred_rel
        
        # RPE에서는 relative error matrix에서 R과 t를 추출
        R_rel_err = rel_err[:3, :3]
        t_rel_err = rel_err[:3, 3]
        
        # Identity와의 차이로 error 계산
        I = np.eye(3)
        rot_err = rotation_error(R_rel_err, I)
        trans_err = translation_error(t_rel_err, np.zeros(3))
        
        rot_errors.append(rot_err)
        trans_errors.append(trans_err)
    
    if len(trans_errors) > 0:
        rpe_trans = np.mean(np.asarray(trans_errors))
        rpe_rot = np.mean(np.asarray(rot_errors))
        print(f"RPE computed from {len(trans_errors)} consecutive frame pairs")
    else:
        rpe_trans = 0.0
        rpe_rot = 0.0
        print(f"⚠️ Warning: No valid frame pairs for RPE computation (only {len(sorted_frames)} frames available)")
    
    return rpe_rot, rpe_trans

def save_scores_to_json(score_json_path, cam_type, ate_rot_err, ate_trans_err, rpe_rot=None, rpe_trans=None, log_message=None):
    """
    Saves or updates rotation and translation errors for a camera type to a JSON file.
    """
    scores_data = {}
    if os.path.exists(score_json_path):
        try:
            with open(score_json_path, 'r') as f:
                scores_data = json.load(f)
        except: # json.JSONDecodeError:
            #print(f"⚠️ Warning: Existing {score_json_path} is empty or malformed. Starting with an empty score dict.")
            scores_data = {}

    score_entry = {
        "ATE_RotErr": np.degrees(ate_rot_err),
        "ATE_TransErr": ate_trans_err,
        "RPE_RotErr": np.degrees(rpe_rot),
        "RPE_TransErr": rpe_trans,
        "log_message": log_message
    }
    
    scores_data[cam_type] = score_entry

    with open(score_json_path, 'w') as f:
        json.dump(scores_data, f, indent=4)
    print(f"✅ Scores for {cam_type} saved to {score_json_path}")

# emjay added ------------------------------------------------
def save_log_to_json(log_json_path, cam_type, log_message):
    """
    Saves or updates error log a camera type to a JSON file.
    """
    scores_data = {}
    if os.path.exists(log_json_path):
        try:
            with open(log_json_path, 'r') as f:
                scores_data = json.load(f)
        except: #json.JSONDecodeError:
            #print(f"⚠️ Warning: Existing {log_json_path} is empty or malformed. Starting with an empty score dict.")
            scores_data = {}

    scores_data[cam_type] = {
        "log_message": log_message
    }

    with open(log_json_path, 'w') as f:
        json.dump(scores_data, f, indent=4)

    print(f"✅ Scores for {cam_type} saved to {log_json_path}")

    return log_message
# ---------------------------------------------------------

def process_cam_vid_combination(cam_vid_args):
    """단일 cam-vid 조합을 처리하는 worker 함수"""
    try:
        cam, vid, path_colmap, gt_json = cam_vid_args
        # if cam == 6 and vid == 28:
        #     pdb.set_trace()
        cam_type = f"cam_type{cam}"
        # cam_type = f"cam{cam:02d}"
        # path_gen_dir = os.path.join(path_gen, f"{cam_type}/frames/video{vid}")
        # path_src_dir = os.path.join(path_src, f"videos/frames/video{vid}")
        # path_project_dir = os.path.join(path_colmap, f"{cam_type}/colmap")                
        path_project_dir = os.path.join(path_colmap, f"{cam_type}/frames")                
        score_json_file_path = os.path.join(path_project_dir, "score.json")# Save scores to JSON

        process_id = f"video{vid}_{cam_type}"
        print(f"🔄 Processing {process_id}")
        
        # Check if input directory exists
        # if not os.path.exists(path_gen_dir):
        #     return save_log_to_json(score_json_file_path, f"video{vid}", f"❌ {process_id}: Input directory not found: {path_gen_dir}")
        
        # COLMAP pipeline 실행 또는 기존 결과 사용
        if not os.path.exists(os.path.join(path_project_dir, f"video{vid}")):
            # try:
            #     # recon_path = run_colmap_pipeline(path_gen_dir, os.path.join(path_project_dir, f"video{vid}"))
            #     recon_path = run_vipe_pipeline(path_gen_dir, os.path.join(path_project_dir, f"video{vid}"))
            # except subprocess.CalledProcessError as e:
            #     return save_log_to_json(score_json_file_path, f"video{vid}", f"❌ {process_id}: VIPE pipeline failed: {str(e)}")            
            return save_log_to_json(score_json_file_path, f"video{vid}", f"❌ {process_id}: missing colmap result")
        else:
            # 실제 COLMAP 실행 대신 미리 생성된 결과 경로 사용
            # recon_path = os.path.join(os.path.join(path_project_dir, f"video{vid}", f"video{vid}", "sparse"), "0")
            # recon_path = os.path.join(os.path.join(path_project_dir, f"video{vid}", "sparse"), "0")
            recon_path = os.path.join(path_project_dir, f"video{vid}")
        
        # recon_path = run_vipe_pipelien(path_gen_dir, os.path.join(path_project_dir, f"video{vid}"))
        images_txt = os.path.join(recon_path, "images.txt")
        
        # Check if images.txt exists
        if not os.path.exists(images_txt):
            return save_log_to_json(score_json_file_path, f"video{vid}", f"❌ {process_id}: images.txt not found: {images_txt}")
        
        # Compute errors
        try:
            ATE_RotErr, ATE_TransErr, RPE_RotErr, RPE_TransErr = compute_errors(images_txt, gt_json, f"cam{cam:02d}")
        except Exception as e:
            return save_log_to_json(score_json_file_path, f"video{vid}", f"❌ {process_id}: Error computing errors: {str(e)}")
        
        try:
            save_scores_to_json(
                score_json_file_path, 
                f"video{vid}", 
                ATE_RotErr, 
                ATE_TransErr, 
                rpe_rot=RPE_RotErr,
                rpe_trans=RPE_TransErr,
                log_message=f"✅ {process_id}: ATE_Rot={ATE_RotErr:.4f}, ATE_Trans={ATE_TransErr:.4f}, RPE_Rot={RPE_RotErr:.4f}, RPE_Trans={RPE_TransErr:.4f}"
            )
            
        except Exception as e:
            return save_log_to_json(score_json_file_path, f"video{vid}", f"❌ {process_id}: Error saving scores: {str(e)}")
        
        return f"✅ {process_id}: ATE_Rot={ATE_RotErr:.4f}, ATE_Trans={ATE_TransErr:.4f}, RPE_Rot={RPE_RotErr:.4f}, RPE_Trans={RPE_TransErr:.4f}"
        
    except Exception as e:
        return print(f"❌ {process_id}: Unexpected error: {str(e)}")

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt_json", type=str, required=True)
    # parser.add_argument("--path_gen", type=str, required=True)
    # parser.add_argument("--path_src", type=str, required=False)
    parser.add_argument("--path_colmap", type=str, required=True)
    parser.add_argument("--num_workers", type=int, default=4, help="Number of CPU workers for multiprocessing")
    parser.add_argument("--sequential", action="store_true", help="Run sequentially instead of multiprocessing")
    parser.add_argument("--is_webvid", action="store_true")
    args = parser.parse_args()
    
    # 모든 cam-vid 조합 생성
    cam_vid_combinations = []
    if args.is_webvid:
        for cam in range(1, 11):
            for vid in range(0, 100):
                # cam_vid_combinations.append((cam, vid, args.path_gen, args.path_src, args.gt_json))
                cam_vid_combinations.append((cam, vid, args.path_colmap, args.gt_json))
    else:  # validation
        for vid in range(0,168):
            json_p = os.path.join(args.gt_json, f"camera_extrinsics_{vid}.json")
            # for cam in range(1, 11):
            for cam in range(1, 2):
                cam_vid_combinations.append((cam, vid, args.path_colmap, json_p))
    
    total_combinations = len(cam_vid_combinations)
    print(f"Found {total_combinations} cam-vid combinations to process")
    print(f"Using {args.num_workers} CPU workers")
    
    if args.sequential:
        print("Running in sequential mode...")
        results = []
        for combination in tqdm(cam_vid_combinations, desc="Processing combinations"):
            result = process_cam_vid_combination(combination)
            results.append(result)
            print(result)
    else:
        try:
            # spawn 컨텍스트 사용 (안정성 향상)
            ctx = mp.get_context('spawn')
            
            with ctx.Pool(processes=args.num_workers) as pool:
                # tqdm으로 진행상황 표시
                results = list(tqdm(
                    pool.imap(process_cam_vid_combination, cam_vid_combinations),
                    total=total_combinations,
                    desc="Processing combinations"
                ))
            
            # 결과 출력
            print("\n=== Processing Results ===")
            success_count = 0
            error_count = 0
            for result in results:
                if result.startswith("✅"):
                    success_count += 1
                else:
                    error_count += 1
                print(result)
            
            print(f"\n=== Summary ===")
            print(f"Total: {total_combinations}")
            print(f"Success: {success_count}")
            print(f"Errors: {error_count}")
                
        except Exception as e:
            print(f"Multiprocessing failed: {e}")
            print("Falling back to sequential processing...")
            
            # 순차 처리 fallback
            results = []
            for combination in tqdm(cam_vid_combinations, desc="Processing combinations (sequential)"):
                result = process_cam_vid_combination(combination)
                results.append(result)
                print(result)

if __name__ == "__main__":
    main()