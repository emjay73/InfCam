#!/bin/bash

GPU_DEVICES="0" 

DATASET_TYPE="webvid"
GT_JSON="cameras/camera_extrinsics_ref40.json"
SEQ_NAME="check vipe_results/vipe and write the subfolder path(relative) of interest."

# convert vipe -> colmap
# write_points3d=false, dump_images=false를 default로 설정해둠.
# seq_name 안쓰면 전부 다 변환.
python scripts/vipe_to_colmap.py vipe_results \
    --sequence ${SEQ_NAME} \
    --skip_existing 
    # --sequence 20250822__with_ktd_fixK_splitval__BS8_ACCUM2_320x544x41_step20000_val_ref40 # 안쓰면 전부 다 변환.

# compute metrics
python run_metric.py \
    --gt_json ${GT_JSON} \
    --path_colmap vipe_results_colmap/${SEQ_NAME} \
    --sequential \
    --is_webvid

# average metrics
python colmap_mean.py --base_folder "vipe_results_colmap/${SEQ_NAME}"
