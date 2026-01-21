#!/bin/bash

GPU_DEVICES="0" 

DATASET_TYPE="val"
GT_JSON="cameras/camera_extrinsics_ref40.json"
SEQ_NAME="GEN3C/RESULT/WEBVID/REF40"

# convert vipe -> colmap
# write_points3d=false, dump_images=false를 default로 설정해둠.
# seq_name 안쓰면 전부 다 변환.
python scripts/vipe_to_colmap.py vipe_results \
    --sequence ${SEQ_NAME} 

# compute metrics
python run_metric.py \
    --gt_json ${GT_JSON} \
    --path_colmap "vipe_results_colmap/${SEQ_NAME}" \
    --sequential 

# average metrics
python colmap_mean.py --base_folder "vipe_results_colmap/${SEQ_NAME}"
