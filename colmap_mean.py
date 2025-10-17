import os
import json
import argparse
import numpy as np
parser = argparse.ArgumentParser()
parser.add_argument("--base_folder", type=str, required=True)
#parser.add_argument("--original_folder", type=str, required=True)
#parser.add_argument("--metadata_path", type=str, required=True)
args = parser.parse_args()

base_dir = args.base_folder #"/home/nas5/hoiyeongjin/repos/Research/nvvs/RESULT/jho/reimple__bs8__step37000"
output_json_path = os.path.join(base_dir, "average_score2.json")

total_ate_rot_err = 0.0
total_ate_trans_err = 0.0
total_rpe_rot_err = 0.0
total_rpe_trans_err = 0.0
count = 0
count_drop = 0
out = []
for cam_id in range(1, 11):
    cam_dir = f"cam_type{cam_id}"
    # score_path = os.path.join(base_dir, cam_dir, "colmap", "score.json")
    score_path = os.path.join(base_dir, cam_dir, "frames", "score.json")
    
    if not os.path.exists(score_path):
        print(f"File not found: {score_path}")
        continue
    else:
        print(f"File found: {score_path}")

    try:
        with open(score_path, "r") as f:
            data = json.load(f)
    except json.JSONDecodeError:
        raise RuntimeError(f"⚠️ Warning: Existing {score_path} is empty or malformed.")

    for video_key, metrics in data.items():

        if ("ATE_TransErr" not in metrics) or ("ATE_RotErr" not in metrics) or ("RPE_TransErr" not in metrics) or ("RPE_RotErr" not in metrics):
            out.append("No ATE_TransErr or ATE_RotErr or RPE_TransErr or RPE_RotErr: "+cam_dir+"_"+video_key)
            count_drop += 1
            continue
        
        if np.isnan(metrics["ATE_TransErr"]) or np.isnan(metrics["ATE_RotErr"]) or np.isnan(metrics["RPE_TransErr"]) or np.isnan(metrics["RPE_RotErr"]):
            out.append("metric is nan: "+cam_dir+"_"+video_key)
            count_drop += 1
            continue

        if metrics["ATE_TransErr"] > 500:
            out.append("ATE_TransErr > 500: "+cam_dir+"_"+video_key)
            count_drop += 1
            continue
        
        if metrics["RPE_TransErr"] > 500:
            out.append("RPE_TransErr > 500: "+cam_dir+"_"+video_key)
            count_drop += 1
            continue
            
        total_ate_rot_err += metrics["ATE_RotErr"]
        total_ate_trans_err += metrics["ATE_TransErr"]
        total_rpe_rot_err += metrics["RPE_RotErr"]
        total_rpe_trans_err += metrics["RPE_TransErr"]
        count += 1

if count == 0:
    print("No valid data found.")
else:
    avg_ate_rot_err = total_ate_rot_err / count
    avg_ate_trans_err = total_ate_trans_err / count
    avg_rpe_rot_err = total_rpe_rot_err / count
    avg_rpe_trans_err = total_rpe_trans_err / count
    avg_result = {
        "AverageATE_RotErr": avg_ate_rot_err,
        "AverageATE_TransErr": avg_ate_trans_err,
        "AverageRPE_RotErr": avg_rpe_rot_err,
        "AverageRPE_TransErr": avg_rpe_trans_err,
        "TotalVideos": count,
        "TotalVideosDrop": count_drop,  
        "out": out
    }

    with open(output_json_path, "w") as out_f:
        json.dump(avg_result, out_f, indent=4)

    print(f"Saved average results to: {output_json_path}")
