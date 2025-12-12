#!/bin/bash

GPU_DEVICES="4" 

DATASET_TYPE="webvid"
GT_JSON="cameras/camera_extrinsics_ref40.json"
#GT_JSON="cameras/camera_extrinsics_ref0.json"

#DATASET_TYPE="validation"
#GT_JSON="DATA/val_webvidFormat/cameras/ref0"

#SEQ_NAME="recam_edit_emjay_jho/results/webvid_lowresol_noaugSyncam_20000step"
#SEQ_NAME="recam_edit_emjay_jho/results/webvid_lowresol_ours_20000step"
SEQ_NAME="GEN3C/RESULT/WEBVID/REF40"

# ref0
#SEQ_NAME="/data/jeonghokim/repos/nvvs/recam_edit_emjay_jho/results/webvidmix_highresol_ours_20000step_ref0_50step"
#SEQ_NAME="rcm/outputs/multicamhy_webvidmix_lowresol_step20000_ref0"

# webvid mix dynamic, ref40
#SEQ_NAME="recam_edit_emjay_jho/results/webvidmix_highresol_ours_93000step_ref40"
#SEQ_NAME="results/trajectorycrafter/webvid_mix_dynamic_ref40"
#SEQ_NAME="results/recammaster/webvidmix_highresol_step20000_ref40"
#SEQ_NAME="results/ours/webvidmix_highresol_ours_20000step_ref40"
#SEQ_NAME="GEN3C/RESULT/WEBVID/REF0"
#SEQ_NAME="GEN3C/RESULT/WEBVID/REF40"
#SEQ_NAME="rcm/outputs/multicamhy_webvidmix_lowresol_step20000_ref40_recam"
#SEQ_NAME="rcm/outputs/multicamhy_webvidmix_lowresol_step20000_ref0"
#SEQ_NAME="/data/jeonghokim/repos/nvvs/rcm/outputs/multicamhy_webvidmix_lowresol_step20000_ref0"
#SEQ_NAME="rcm/outputs/multicamhy_webvidmix_lowresol_step20000_ref40_recam"

# val
# SEQ_NAME="trajectorycrafter/val_highresol/webvidFormat"
# SEQ_NAME="recam_edit_emjay_jho/results/val_highresol_ours_20000step"
# SEQ_NAME="recam_edit_emjay_jho/results/val_highresol_ours_93000step"

# convert vipe -> colmap
# write_points3d=false, dump_images=false를 default로 설정해둠.
# seq_name 안쓰면 전부 다 변환.
python scripts/vipe_to_colmap.py vipe_results \
    --sequence ${SEQ_NAME} 
    # --sequence 20250822__with_ktd_fixK_splitval__BS8_ACCUM2_320x544x41_step20000_val_ref40 # 안쓰면 전부 다 변환.

# compute metrics
if [ "$DATASET_TYPE" = "webvid" ]; then
     
    python run_metric.py \
        --gt_json ${GT_JSON} \
        --path_colmap vipe_results_colmap/${SEQ_NAME} \
        --sequential \
        --is_webvid

else  # validation
    python run_metric.py \
        --gt_json ${GT_JSON} \
        --path_colmap "vipe_results_colmap/${SEQ_NAME}" \
        --sequential 
#ex) conda activate vipe
#ex) python -m pdb run_metric.py --path_colmap vipe_results_colmap/trajectorycrafter/val_highresol/webvidFormat --sequential
fi

# average metrics
python colmap_mean.py --base_folder "vipe_results_colmap/${SEQ_NAME}"
