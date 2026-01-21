#!/bin/bash

# how to:
# bash run_vipe.sh

GPU_DEVICES="0" # multi gpu not supported�. only one gpu
CAM_TYPES=(1 2 3 4 5 6 7 8 9 10)

DATASET_TYPE='val'
PATH_SRC_DIR="./DATA/AugMCV/val"
PATH_GEN_DIR="path_to_results_dir"

# parse video
# transform to webvid format
python val_to_webvidFormat.py --video_dir "${PATH_SRC_DIR}" --is_gt
python val_to_webvidFormat.py --video_dir "${PATH_GEN_DIR}"
python video2frame_hierarchy_val.py --parse_dir "${PATH_SRC_DIR}_webvidFormat"
python video2frame_hierarchy_val.py --parse_dir "${PATH_GEN_DIR}/webvidFormat" --save_first_frame --src_dir "${PATH_SRC_DIR}_webvidFormat" 
PATH_GEN_DIR="${PATH_GEN_DIR}/webvidFormat"

# 동적으로 경로 생성
PATH_GEN_CAM_DIRS="["
for i in "${!CAM_TYPES[@]}"; do
    if [ $i -gt 0 ]; then
        PATH_GEN_CAM_DIRS+=","
    fi
    PATH_GEN_CAM_DIRS+="'${PATH_GEN_DIR}/cam_type${CAM_TYPES[$i]}/frames'"

    #num="${CAM_TYPES[$i]}"
    #cam_id="$(printf '%02d' "$num")"
    #PATH_GEN_CAM_DIRS+="'${PATH_GEN_DIR}/cam${cam_id}/frames'"
done
PATH_GEN_CAM_DIRS+="]"

echo "Generated PATH_GEN_CAM_DIRS: $PATH_GEN_CAM_DIRS"

# run vipe
CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python run_vipe.py \
    pipeline.output.skip_exists=false \
    pipeline.output.save_viz=false \
    pipeline.post.depth_align_model=null \
    streams.base_path="${PATH_GEN_CAM_DIRS}" \
    streams.name_path_parts=6 # base_path 뒤에서부터 몇번째 폴더 이름까지를 저장할 때 사용할지. base_path[-name_path_parts:]
    
    
