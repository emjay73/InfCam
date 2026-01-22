#!/bin/sh
mkdir -p exp_nohup
#SBATCH -J first_run
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --time=47:59:00
#SBATCH -p amd_a100nv_8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --comment pytorch
#SBATCH -o exp_nohup/%j_%x.txt
#SBATCH -e exp_nohup/%j_%x.err

BS=1 ACCUM=5 LR="1e-4"
GPU_IDX="0,1,2,3,4,5,6,7" N_GPUS=$(echo "$GPU_IDX" | awk -F',' '{print NF}')
CUDA_VISIBLE_DEVICES=${GPU_IDX} python train_recammaster.py --task train \
 --dataset_path "./DATA/MultiCamVideo-Dataset/MultiCamVideo-Dataset" \
 --dit_path "models/Wan-AI/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors" \
 --train_batch_size ${BS} --accumulate_grad_batches ${ACCUM} --learning_rate ${LR} \
 --steps_per_epoch 8000 --use_gradient_checkpointing \
 --output_path ./logs/bs${BS}_accum${ACCUM}_gpu${N_GPUS}_lr${LR}