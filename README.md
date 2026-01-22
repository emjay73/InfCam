VAE Feature Extraction for LDM Training (ReCamMaster-based)

This repository contains data preprocessing code to pre-extract VAE latent features
for training a Latent Diffusion Model (LDM), based on the official ReCamMaster framework.

The goal is to convert multi-camera video data into VAE latent representations
before LDM training, reducing training cost and simplifying the pipeline.

------------------------------------------------------------
1. Environment Setup
------------------------------------------------------------

The environment setup follows the official ReCamMaster repository.
Please ensure CUDA and GPU drivers are properly installed.

1.1 Create Conda Environment

conda create -n recammaster python=3.9 -y
conda activate recammaster

1.2 Install PyTorch

Install PyTorch according to your CUDA version.

Example (CUDA 11.8):

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

1.3 Install Dependencies
pip install -r requirements.txt

Main dependencies include:
- numpy
- opencv-python
- einops
- tqdm
- decord
- transformers
- diffusers

------------------------------------------------------------
2. Pretrained Model Preparation
------------------------------------------------------------

ReCamMaster uses pretrained components from the Wan2.1 Text-to-Video model.
All required models must be downloaded in advance and placed in the correct paths.

2.1 Required Pretrained Models

- Wan2.1 VAE (video VAE for latent encoding)
- UMT5 / T5 text encoder used in Wan2.1

Please download these models following the instructions in the official
ReCamMaster repository.

Notes:
- Model paths are explicitly specified via command-line arguments.
- Do not rename model files unless you also update the corresponding paths.

------------------------------------------------------------
3. Dataset Preparation
------------------------------------------------------------

Videos are prepared in the same format as the "samples" directory.
Captions are generated following the procedure described in:
https://github.com/LLaVA-VL/LLaVA-NeXT

The finalized metadata file can be found at:
./samples/metadata.csv

------------------------------------------------------------
4. VAE Feature Extraction
------------------------------------------------------------

The following command extracts VAE latent features from example videos.
This is intended to be run BEFORE LDM training as a data preprocessing step.

4.1 Example Command

CUDA_VISIBLE_DEVICES=0 python train_recammaster.py --task data_process \
  --dataset_path "./samples" \
  --metadata_file_name "metadata.csv" \
  --text_encoder_path "models/Wan-AI/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth" \
  --vae_path "models/Wan-AI/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth" \
  --num_frames 81 \
  --height 480 \
  --width 832 \
  --dataloader_num_workers 2

4.2 Argument Description

--task data_process
    Runs VAE feature extraction instead of training.

--dataset_path
    Path to the video dataset root directory.

--metadata_file_name
    CSV file containing dataset metadata.

--text_encoder_path
    Path to the pretrained text encoder.

--vae_path
    Path to the pretrained VAE model.

--num_frames
    Number of frames per video clip.

--height, --width
    Spatial resolution of input frames.

--dataloader_num_workers
    Number of workers for data loading.

------------------------------------------------------------
5. Output
------------------------------------------------------------

- Input videos are converted into VAE latent tensors.
- The extracted latents can be directly used for Latent Diffusion Model training.
- This preprocessing step significantly reduces computational cost during LDM training.

------------------------------------------------------------
6. Acknowledgement
------------------------------------------------------------

This project is based on the ReCamMaster(https://github.com/KlingTeam/ReCamMaster) framework.
All original code and pretrained models belong to their respective authors.
