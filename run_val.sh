#!/bin/bash
SEED=0

# for CAM in {1..10}; do
for CAM in {9..10}; do

    CUDA_VISIBLE_DEVICES=7 python inference_val.py \
        --cam_type ${CAM} \
        --ckpt_path "models/InfCam/step20000.ckpt" \
        --output_dir "./results/val/warp_perframe_step20k" \
        --dataset_path "/home/emjay_data/AugMCV/val" \
        --metadata_file_name "metadata_static_video.csv" \
        --num_frames 81 --width 832 --height 480 \
        --num_inference_steps 20 \
        --seed ${SEED}

done
