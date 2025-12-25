# InfCam: Infinite-Homography as Robust Conditioning for Camera-Controlled Video Generation

<div align="center">
<div align="center" style="margin-top: 0px; margin-bottom: 0px;">
<!-- <img src=https://github.com/user-attachments/assets/81ccf80e-f4b6-4a3d-b47a-e9c2ce14e34f width="30%"/> -->
<!-- <img src=assets/infcam_icon.png width="30%"/> -->
</div>

### [<a href="https://arxiv.org/abs/2512.17040" target="_blank">arXiv</a>] [<a href="https://emjay73.github.io/InfCam/" target="_blank">Project Page</a>] 
<!-- [<a href="" target="_blank">Dataset(Coming Soon)</a>] -->
[Min-Jung Kim<sup>*</sup>](https://emjay73.github.io/), [Jeongho Kim<sup>*</sup>](https://scholar.google.co.kr/citations?user=4SCCBFwAAAAJ&hl=ko), [Hoiyeong Jin<sup></sup>](https://scholar.google.co.kr/citations?hl=ko&user=Jp-zhtUAAAAJ), [Junha Hyung<sup></sup>](https://junhahyung.github.io/), [Jaegul Choo<sup></sup>](https://sites.google.com/site/jaegulchoo)
<br>
*Equal Contribution
<p align="center">
  <img src="assets/GSAI_preview_image.png" width="20%" alt="GSAI Preview">
</p>

<!-- <sup></sup>KAIST AI -->

</div>

## <img src="https://img.icons8.com/fluency/48/video-playlist.png" width="28" style="vertical-align:middle;"/> Teaser Video
https://github.com/user-attachments/assets/1c52baf4-b5ff-417e-a6c6-c8570e667bd8

## 🔥 Updates
- [ ] Release data augmentation code 
- [x] Release training code (2025.12.26)
- [x] Release inference code (2025.12.19)
- [x] Release model weights (2025.12.19)

  
## 📖 Introduction

**TL;DR:** Given a video and a target camera trajectory, InfCam generates a video that faithfully follows the specified camera path without depth prior. <br>


## ⚙️ Code

### Environment 

```
conda create -n infcam python=3.12
conda activate infcam

# for inference only
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install cupy-cuda12x
pip install transformers==4.46.2
pip install sentencepiece
pip install controlnet-aux==0.0.7
pip install imageio
pip install imageio[ffmpeg]
pip install safetensors
pip install einops
pip install protobuf
pip install modelscope
pip install ftfy
pip install lpips
pip install lightning
pip install pandas
pip install matplotlib
pip install wandb
pip install ffmpeg-python
pip install numpy
pip install opencv-python

# for training
pip install deepspeed
```

### 🕹️ Inference

Hardware: 1x NVIDIA H100 80GB GPUs.  \
Memory Usage: > 50 GB (48G of memory for UniDepth and 28 GB for the InfCam pipeline.)

Step 1: Download the pretrained checkpoints

(1) Pre-trained Wan2.1 model

```shell
python download_wan2.1.py
```

(2) Pre-trained UniDepth model 

Download the pre-trained weights from [huggingface](https://huggingface.co/lpiccinelli/unidepth-v2-vitl14) and place it in ```models/unidepth-v2-vitl14```.
```shell
cd models
git clone https://huggingface.co/lpiccinelli/unidepth-v2-vitl14
```

(3) Pre-trained InfCam checkpoint

Download the pre-trained InfCam weights from [huggingface](https://huggingface.co/emjay73/InfCam/tree/main) and place it in ```models/InfCam```.

```shell
cd models
git clone https://huggingface.co/emjay73/InfCam
```
Step 2: Test the example videos

```shell
bash run_inference.sh
```
or 
```shell
for CAM in {1..10}; do
    CUDA_VISIBLE_DEVICES=0 python inference_infcam.py \
        --cam_type ${CAM} \
	    --ckpt_path "models/InfCam/step35000.ckpt" \
        --camera_extrinsics_path "./sample_data/cameras/camera_extrinsics_10types.json" \
        --output_dir "./results/sample_data" \
	    --dataset_path "./sample_data" \
        --metadata_file_name "metadata.csv" \
        --num_frames 81 --width 832 --height 480 \
        --num_inference_steps 20 \
	    --zoom_factor 1.0 \
        --k_from_unidepth \
	    --seed ${SEED}
done
```

Step 3: Test your own videos

If you want to test your own videos, you need to prepare your test data following the structure of the ```sample_data``` folder. This includes N mp4 videos, each with at least 81 frames, and a ```metadata.csv``` file that stores their paths and corresponding captions. You can refer to the '[caption branch](https://github.com/emjay73/InfCam/tree/caption) for metadata.csv extraction.


We provide several preset camera types, as shown in the table below.
These follow the [ReCamMaster](https://jianhongbai.github.io/ReCamMaster/) presets, but the starting point of each trajectory differs from that of the initial frame.

| cam_type       | Trajectory                  |
|-------------------|-----------------------------|
| 1 | Pan Right                   |
| 2 | Pan Left                    |
| 3 | Tilt Up                     |
| 4 | Tilt Down                   |
| 5 | Zoom In                     |
| 6 | Zoom Out                    |
| 7 | Translate Up (with rotation)   |
| 8 | Translate Down (with rotation) |
| 9 | Arc Left (with rotation)    |
| 10 | Arc Right (with rotation)   |

### 🚂 Train
Hardware: 4x NVIDIA H100 80GB GPUs.

Memory Usage(low resolution, B=8, F=41 H=320 W=544): Approximately 52GB of VRAM per GPU during training.

Memory Usage(high resolution, B=2, F=81 H=480 W=832): Approximately 56GB of VRAM per GPU during training. 

Step1. Prepare Dataset

Prepare dataset by applying data augmentation to the [MultiCamVideo-Dataset](https://huggingface.co/datasets/KlingTeam/MultiCamVideo-Dataset).

If you would like to see an example of the training set with augmentation already applied, you can download a subset from Hugging Face: [AugMCV](https://huggingface.co/datasets/emjay73/AugMCV).
```
mkdir DATA
cd DATA

# download train data subset
git clone https://huggingface.co/datasets/emjay73/AugMCV

cd AugMCV
tar -xvzf AugMCV.tar.gz --strip-components=1
```

The training data should follow the directory structure shown below:
```
InfCam
└── DATA
    └── AugMCV
        ├── train
        │   ├── f18_aperture10
        │   │   └── scene1_0
        │   │       ├── cameras
        │   │       │   ├── camera_extrinsics.json
        │   │       │   └── video_mapping.json
        │   │       └── videos
        │   │           ├── cam01.mp4
        │   │           ├── cam01.mp4.[config].pth
        │   │           ...
        │   │           ├── cam10.mp4
        │   │           └── cam10.mp4.[config].pth
        │   ├── f18_aperture10_aug
        │   ...
        │   └── f50_aperture2.4
        ├── test
        ├── metadata_train_aug.csv
        └── metadata_test_aug.csv

```

Step2. Run train
```
bash run_train.sh
```

## 🤗 Special Thanks
We build upon the following repositories and thank the authors for their incredible work:

[ReCamMaster](https://jianhongbai.github.io/ReCamMaster/): Re-capture in-the-wild videos with novel camera trajectories, and release a multi-camera synchronized video dataset rendered with Unreal Engine 5.

[WAN2.1](https://github.com/Wan-Video/Wan2.1): A comprehensive and open suite of video foundation models.

[UniDepthV2](https://github.com/lpiccinelli-eth/UniDepth): Monocular metric depth estimation.

## 🌟 Citation

Please leave us a star 🌟 and cite our paper if you find our work helpful.
```
bibtex
@article{kim2025infcam,
  title={Infinite-Homography as Robust Conditioning for Camera-Controlled Video Generation},
  author={Kim, Min-Jung and Kim, Jeongho and Jin, Hoiyeong and Hyung, Junha and Choo, Jaegul},
  journal={arXiv preprint arXiv:2512.17040},
  year={2025}
}
```
