import argparse
import os
from os.path import join as opj
from glob import glob
from tqdm import tqdm

# pip install git+https://github.com/LLaVA-VL/LLaVA-NeXT.git
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle
from PIL import Image
import requests
import copy
import torch
import sys
import warnings
from decord import VideoReader, cpu
import numpy as np
warnings.filterwarnings("ignore")

def parse_args():
    parser = argparse.ArgumentParser(description="Hacked Inference for LLaVA-Video")
    parser.add_argument("--dataset_dir", type=str, default=None)
    parser.add_argument("--only_first_cam", action="store_true", help="Only process the first camera of each video.")
    parser.add_argument("--from_back", action="store_true")
    parser.add_argument("--n_proc", type=int, default=1)
    parser.add_argument("--proc_idx", type=int, default=0)
    parser.add_argument("--save_root_dir", type=str, required=True, help="ex) ./results")
    args = parser.parse_args()
    return args

def split_procidx(lst, n_proc, proc_idx):
    len_ps = len(lst)
    if len_ps % n_proc == 0:
        n_infer = len_ps // n_proc
    else:
        n_infer = len_ps // n_proc + 1
    
    start_idx = int(proc_idx * n_infer)
    end_idx = start_idx + n_infer
    sub_lst = lst[start_idx:end_idx]
    return sub_lst

def load_video(video_path, max_frames_num,fps=1,force_sample=False):
    if max_frames_num == 0:
        return np.zeros((1, 336, 336, 3))
    vr = VideoReader(video_path, ctx=cpu(0),num_threads=1)
    total_frame_num = len(vr)
    video_time = total_frame_num / vr.get_avg_fps()
    fps = round(vr.get_avg_fps()/fps)
    frame_idx = [i for i in range(0, len(vr), fps)]
    frame_time = [i/fps for i in frame_idx]
    if len(frame_idx) > max_frames_num or force_sample:
        sample_fps = max_frames_num
        uniform_sampled_frames = np.linspace(0, total_frame_num - 1, sample_fps, dtype=int)
        frame_idx = uniform_sampled_frames.tolist()
        frame_time = [i/vr.get_avg_fps() for i in frame_idx]
    frame_time = ",".join([f"{i:.2f}s" for i in frame_time])
    spare_frames = vr.get_batch(frame_idx).asnumpy()
    # import pdb;pdb.set_trace()
    return spare_frames,frame_time,video_time
pretrained = "lmms-lab/LLaVA-Video-7B-Qwen2"
model_name = "llava_qwen"
device = "cuda"
device_map = "auto"
tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained, None, model_name, torch_dtype="bfloat16", device_map=device_map)  # Add any other thing you want to pass in llava_model_args
model.eval()

args = parse_args()
video_ps = sorted(glob(opj(args.dataset_dir, "*.mp4")))
if args.from_back:
    video_ps = split_procidx(video_ps, args.n_proc, args.proc_idx)[::-1]
else:
    video_ps = split_procidx(video_ps, args.n_proc, args.proc_idx)
print(f"Process {args.proc_idx} in {args.n_proc} processes, processing {len(video_ps)} videos.")
output = []
for video_path in tqdm(video_ps, total=len(video_ps)):
    save_path = opj(args.save_root_dir, "/".join(video_path.split("/")[-1:]).replace(".mp4", ".txt"))
    if os.path.exists(save_path):
        print(f"Skip {video_path}, already exists.")
        continue

    max_frames_num = 64
    video,frame_time,video_time = load_video(video_path, max_frames_num, 1, force_sample=True)
    video = image_processor.preprocess(video, return_tensors="pt")["pixel_values"].cuda().bfloat16()
    video = [video]
    conv_template = "qwen_1_5"  # Make sure you use correct chat template for different models
    time_instruciton = f"The video lasts for {video_time:.2f} seconds, and {len(video[0])} frames are uniformly sampled from it. These frames are located at {frame_time}.Please answer the following questions related to this video."
    question = DEFAULT_IMAGE_TOKEN + f"\n{time_instruciton}\nPlease describe this video in detail."
    conv = copy.deepcopy(conv_templates[conv_template])
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    prompt_question = conv.get_prompt()
    input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
    
    cont = model.generate(
        input_ids,
        images=video,
        modalities= ["video"],
        do_sample=False,
        temperature=0,
        max_new_tokens=300,
    )
    text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)[0].strip()
    print(len(text_outputs.split(" ")))
    ## save
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(text_outputs)
