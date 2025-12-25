#!/bin/bash

BS=8 ACCUM=1 F=41 H=320 W=544 # low resolution training
# BS=2 ACCUM=1 F=81 H=480 W=832 # high resolution training

SAVE_FREQ=500
TAG="train_${F}x${H}x${W}"
TIME=$(TZ="Asia/Seoul" date +%Y%m%d)

# CUDA_VISIBLE_DEVICES=0 python -m pdb train_infcam.py \
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_infcam.py \
    --task train  \
    --dataset_path "./DATA/AugMCV" \
    --batch_size ${BS} --accumulate_grad_batches ${ACCUM} --learning_rate 1e-4 \
    --num_frames ${F} --height ${H} --width ${W} \
    --metadata_file_name "metadata_train_aug_subset.csv" \
    --save_every_n_steps ${SAVE_FREQ} \
    --output_path "./models/train" \
    --run_name "${TIME}__${TAG}__BS${BS}_ACCUM${ACCUM}_${H}x${W}x${F}" \
    --use_gradient_checkpointing 

    # optional arguments
    # --use_wandb \
    # --resume_ckpt_path "path_to_ckpt.ckpt"

