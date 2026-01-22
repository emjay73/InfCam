#### data process
## MulticamVideo
CUDA_VISIBLE_DEVICES=3 python train_recammaster.py --task data_process \
 --dataset_path "./DATA/MultiCamVideo-Dataset-hy" --metadata_file_name "metadata_train_aug.csv" \
 --text_encoder_path "models/Wan-AI/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth" --vae_path "models/Wan-AI/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth" \
 --num_frames 81 --height 480 --width 832 --dataloader_num_workers 2 \
 --data_process_n_proc 8 --data_process_proc_idx 7

## DL3DV
CUDA_VISIBLE_DEVICES=7 python train_recammaster.py --task data_process \
 --dataset_path "./DATA/DL3DV" --metadata_file_name "metadata.csv" \
 --text_encoder_path "models/Wan-AI/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth" --vae_path "models/Wan-AI/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth" \
 --num_frames 41 --height 320 --width 544 --dataloader_num_workers 2 \
 --data_process_use_split --data_process_split_stride 24 \
 --data_process_n_proc 12 --data_process_proc_idx 10

## Ego4D
CUDA_VISIBLE_DEVICES=1 python train_recammaster.py --task data_process \
 --dataset_path "./DATA/Ego4D" --metadata_file_name "metadata.csv" \
 --text_encoder_path "models/Wan-AI/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth" --vae_path "models/Wan-AI/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth" \
 --num_frames 41 --height 320 --width 544 --dataloader_num_workers 2 \
 --data_process_use_split --data_process_split_stride 96


#### train
BS=1 ACCUM=4 LR=1e-4
F=41 H=320 W=544
TAG="multicamhy"
GPU_IDX="4,5,6,7"

N_GPUS=$(echo "$GPU_IDX" | awk -F',' '{print NF}') TIME=$(TZ="Asia/Seoul" date +%Y%m%d)
CUDA_VISIBLE_DEVICES=${GPU_IDX} python train_recammaster.py --task train \
 --dataset_path "./DATA/MultiCamVideo-Dataset-hy" --metadata_file_name metadata_train.csv \
 --num_frames ${F} --height ${H} --width ${W} \
 --train_batch_size ${BS} --accumulate_grad_batches ${ACCUM} --learning_rate ${LR} --dataloader_num_workers 0 \
 --output_path ./logs/${TIME}__${TAG}__bs${BS}_accum${ACCUM}_gpu${N_GPUS}_lr${LR}_${F}x${H}x${W}
# --dataset_path "./DATA/DL3DV" --metadata_file_name metadata_stride24_lowresol.csv \
# --dataset_path2 "./DATA/Ego4D" --metadata_file_name2 metadata_stride96_lowresol.csv \
# --use_gradient_checkpointing \
# --use_reverse \
# --no_projector \
# --use_src_adapter \
# --resume_ckpt_path "./logs/bs4_accum1_gpu8_lr1e-4_81x480x832/lightning_logs/version_0/checkpoints/step1750.ckpt" \


## inference
for i in {1..10}; do
    CUDA_VISIBLE_DEVICES=7 python inference_recammaster.py --cam_type ${i} \
    --ckpt_path "./logs/20251107__multicamhy__bs1_accum4_gpu2_lr1e-4_41x320x544/lightning_logs/version_0/checkpoints/step20000.ckpt" \
    --dataset_path "./DATA/MultiCamVideo-Dataset/webvid_mix_dynamic" \
    --num_frames 41 --height 320 --width 544 \
    --tgt_cam_path "./example_test_data/cameras/camera_extrinsics_ref40.json" \
    --output_dir "./outputs/multicamhy_webvidmix_lowresol_step20000_moreBS_ref40"
done

# --num_frames 41 --height 320 --width 544 \
# --dataset_path "./DATA/MultiCamVideo-Dataset/webvid_test_data"
# --ckpt_path "./models/ReCamMaster/checkpoints/step20000.ckpt" \
# --use_reverse \
# --no_projector \
# --repeat_time_rope \
# --repeat_first_frame 8 \
# --use_src_adapter \
# --use_interval_pose \
# --src_cam_path "./example_test_data/cameras/left45.txt" \
# --tgt_cam_path "./example_test_data/cameras/camera_extrinsics_ref40_recam.json" --repeat_first_frame 8 \  # for first frame change
# --ckpt_path "./logs/20250613__bs1_accum4_gpu2_lr1e-4_41x320x544/lightning_logs/version_0/checkpoints/step20000.ckpt" \


# inference val
for CAM in {1..10}; do    
CUDA_VISIBLE_DEVICES=7 python inference_val.py \
    --cam_type ${CAM} --lora_rank 0 --lora_alpha 64 \
    --idx_modified_layers {0..29} \
    --ckpt_path "./logs/20251117__multicamhy__bs1_accum4_gpu4_lr1e-4_41x320x544/lightning_logs/version_0/checkpoints/step30000.ckpt" \
    --output_dir "./outputs/multicamhy_val_lowresol_30000" \
    --dataset_path "./DATA/MultiCamVideo-Dataset-hy/val" \
    --metadata_file_name "metadata_static_video.csv" \
    --num_frames 41 --height 320 --width 544
done
# --num_frames 81 --height 480 --width 832
# --ckpt_path "./logs/20250613_reimplement/checkpoints/step20000.ckpt" \


# inference teaser
for i in {1..10}; do
    CUDA_VISIBLE_DEVICES=4 python inference_recammaster.py --cam_type ${i} \
    --ckpt_path "./models/ReCamMaster/checkpoints/step20000.ckpt" \
    --dataset_path "./DATA/nvvs_teaser" \
    --num_frames 81 --height 480 --width 832 \
    --tgt_cam_path "./DATA/nvvs_teaser/cameras/camera_extrinsics_ref40.json" \
    --output_dir "./outputs/teaser_highresol_step20000_ref40"
done