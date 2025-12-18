# InfCam: Infinite-Homography as Robust Conditioning for Camera-Controlled Video Generation

<div align="center">
<div align="center" style="margin-top: 0px; margin-bottom: 0px;">
<!-- <img src=https://github.com/user-attachments/assets/81ccf80e-f4b6-4a3d-b47a-e9c2ce14e34f width="30%"/> -->
<!-- <img src=assets/infcam_icon.png width="30%"/> -->
</div>

### [<a href="https://arxiv.org/" target="_blank">arXiv(Coming Soon)</a>] [<a href="https://emjay73.github.io/InfCam/" target="_blank">Project Page</a>] 
<!-- [<a href="" target="_blank">Dataset(Coming Soon)</a>] -->
[Min-Jung Kim<sup>*</sup>](https://emjay73.github.io/), [Jeongho Kim<sup>*</sup>](https://scholar.google.co.kr/citations?user=4SCCBFwAAAAJ&hl=ko), [Hoiyeong Jin<sup></sup>](https://scholar.google.co.kr/citations?hl=ko&user=Jp-zhtUAAAAJ), [Junha Hyung<sup></sup>](https://junhahyung.github.io/), [Jaegul Choo<sup></sup>](https://sites.google.com/site/jaegulchoo)
<br>
*Equal Contribution
<p align="center">
  <img src="assets/GSAI_preview_image.png" width="60%" alt="GSAI Preview">
</p>

<sup></sup>KAIST AI

</div>

## <img src="https://img.icons8.com/fluency/48/video-playlist.png" width="28" style="vertical-align:middle;"/> Teaser Video
<p align="center">
  <video src="assets/ours_grid_in-the-wild_1.mp4" autoplay loop muted playsinline style="max-width: 90%; height: auto; border-radius: 12px; box-shadow: 0 2px 20px rgba(0,0,0,0.15);" poster="assets/GSAI_preview_image.png"></video>
</p>

## 🔥 Updates
- [ ] Release training code
- [ ] Release inference code and model weights

  
## 📖 Introduction

**TL;DR:** Given a video and a target camera trajectory, InfCam generates a video that faithfully follows the specified camera path without depth prior. <br>


## ⚙️ Code
### Inference
Step 1: Set up the environment

```
conda create -n infcam python=3.12
conda activate infcam

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
```

Step 2: Download the pretrained checkpoints
1. Download the pre-trained Wan2.1 model

```shell
python download_wan2.1.py
```

2. Download the pre-trained UniDepth model 

Download the pre-trained weights from [huggingface](https://huggingface.co/lpiccinelli/unidepth-v2-vitl14) and place it in ```models/unidepth-v2-vitl14```.
```shell
cd models
git clone https://huggingface.co/lpiccinelli/unidepth-v2-vitl14
```

3. Download the pre-trained InfCam checkpoint

Download the pre-trained InfCam weights from [huggingface](https://huggingface.co/emjay73/InfCam/tree/main) and place it in ```models/InfCam```.

```shell
cd models
git clone https://huggingface.co/emjay73/InfCam
```
Step 3: Test the example videos

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
This inference code requires 48 GB of memory for UniDepth and 28 GB for the InfCam pipeline.

Step 4: Test your own videos

If you want to test your own videos, you need to prepare your test data following the structure of the ```sample_data``` folder. This includes N mp4 videos, each with at least 81 frames, and a ```metadata.csv``` file that stores their paths and corresponding captions. You can refer to the '[caption branch](https://github.com/emjay73/InfCam/tree/caption) for metadata.csv extraction.


We provide several preset camera types, as shown in the table below.
These follow the ReCamMaster presets, but the starting point of each trajectory differs from that of the initial frame.

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


## 🤗 Thank You Note
Our work is based on the following repositories.\
Thank you for your outstanding contributions!

[ReCamMaster](https://jianhongbai.github.io/ReCamMaster/): Re-capture in-the-wild videos with novel camera trajectories, and release a multi-camera synchronized video dataset rendered with Unreal Engine 5.

[WAN2.1](https://github.com/Wan-Video/Wan2.1): A comprehensive and open suite of video foundation models.

[UniDepthV2](https://github.com/lpiccinelli-eth/UniDepth): Monocular metric depth estimation.

## 🌟 Citation

Please leave us a star 🌟 and cite our paper if you find our work helpful.
```

```
