import copy
import os
import re
import torch, os, imageio, argparse
from torchvision.transforms import v2
from einops import rearrange
import lightning as pl
import pandas as pd
from diffsynth import WanVideoReCamMasterPipeline, ModelManager, load_state_dict
import torchvision
from PIL import Image
import numpy as np
import random
import json
import torch.nn as nn
import torch.nn.functional as F
import shutil

#### jho added >>>>
from os.path import join as opj
from glob import glob
from tqdm import tqdm
from pytorch_lightning.utilities import rank_zero_info, rank_zero_only
from pytorch_lightning import seed_everything
from utils import AverageMeter
from torch.utils.data import ConcatDataset, WeightedRandomSampler
from ipdb import set_trace as breakpoint
seed_everything(42)
#### jho added <<<<

class TextVideoDataset(torch.utils.data.Dataset):
    def __init__(self, base_path, metadata_path, max_num_frames=81, frame_interval=1, num_frames=81, height=480, width=832, is_i2v=False, n_proc=1, proc_idx=0, use_reverse=False, use_split=False, split_stride=None):
        metadata = pd.read_csv(metadata_path)
        if "MultiCamVideo-Dataset" in metadata_path:
            folder_name = "train"
        elif "DL3DV" in metadata_path:
            folder_name = "DL3DV-ALL-960P_unzip_eachK"
        elif "Ego4D" in metadata_path:
            folder_name = "takes"
        else:
            folder_name = "videos"

        self.path = [os.path.join(base_path, folder_name, file_name) for file_name in metadata["file_name"]]
        self.text = metadata["text"].to_list()
        
        #### jho added: filtering and split >>>>
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
        
        if n_proc != 1:
            self.path = split_procidx(self.path, n_proc, proc_idx)
            self.text = split_procidx(self.text, n_proc, proc_idx)

        filtered_path = []
        filtered_text = []
        data_prefix = f"{num_frames}x{height}x{width}_" if not (num_frames == 81 and height == 480 and width == 832) else ""
        if use_reverse:
            data_prefix += "reverse_"
        for p, t in zip(self.path, self.text):
            if use_split:
                tensor_dir = os.path.splitext(p)[0]
                if not len(glob(opj(tensor_dir, "*.pth"))):
                    filtered_path.append(p)
                    filtered_text.append(t)
            else:
                tensor_pth = p + f".{data_prefix}tensors.pth"
                if not os.path.exists(tensor_pth):
                    filtered_path.append(p)
                    filtered_text.append(t)
        self.path = filtered_path
        self.text = filtered_text
        rank_zero_info(f"{len(self.path)} to process\n"*4)
        #### jho added <<<<

        
        self.max_num_frames = max_num_frames
        self.frame_interval = frame_interval
        self.num_frames = num_frames
        self.height = height
        self.width = width
        self.is_i2v = is_i2v
            
        self.frame_process = v2.Compose([
            v2.CenterCrop(size=(height, width)),
            v2.Resize(size=(height, width), antialias=True),
            v2.ToTensor(),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        
        self.use_reverse = use_reverse
        self.use_split = use_split
        self.split_stride = split_stride
        
    def crop_and_resize(self, image):
        width, height = image.size
        scale = max(self.width / width, self.height / height)
        image = torchvision.transforms.functional.resize(
            image,
            (round(height*scale), round(width*scale)),
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR
        )
        return image


    def load_frames_using_imageio(self, file_path, max_num_frames, start_frame_id, interval, num_frames, frame_process):
        reader = imageio.get_reader(file_path)

        # if reader.count_frames() < max_num_frames:
        #     reader.close()
        #     return None

        all_frames = [reader.get_data(i) for i in range(reader.count_frames())]
        
        if self.use_reverse:
            start_frame_id = reader.count_frames() - 1 - start_frame_id
            interval = -interval
        
        frames = []
        first_frame = None
        if self.use_split:
            target_num_frames = min(len(all_frames), 10000)
        else:
            target_num_frames = num_frames
        for frame_id in range(target_num_frames):
            frame = all_frames[start_frame_id + frame_id * interval]
            frame = Image.fromarray(frame)
            frame = self.crop_and_resize(frame)
            if first_frame is None:
                first_frame = np.array(frame)
            frame = frame_process(frame)
            frames.append(frame)
        reader.close()

        frames = torch.stack(frames, dim=0)
        frames = rearrange(frames, "T C H W -> C T H W")

        if self.is_i2v:
            return frames, first_frame
        else:
            return frames


    def load_video(self, file_path):
        start_frame_id = 0
        frames = self.load_frames_using_imageio(file_path, self.max_num_frames, start_frame_id, self.frame_interval, self.num_frames, self.frame_process)
        return frames
    
    
    def is_image(self, file_path):
        file_ext_name = file_path.split(".")[-1]
        if file_ext_name.lower() in ["jpg", "jpeg", "png", "webp"]:
            return True
        return False
    
    
    def load_image(self, file_path):
        frame = Image.open(file_path).convert("RGB")
        frame = self.crop_and_resize(frame)
        first_frame = frame
        frame = self.frame_process(frame)
        frame = rearrange(frame, "C H W -> C 1 H W")
        return frame


    def __getitem__(self, data_id):
        text = self.text[data_id]
        path = self.path[data_id]
        while True:
            try:
                if self.is_image(path):
                    if self.is_i2v:
                        raise ValueError(f"{path} is not a video. I2V model doesn't support image-to-image training.")
                    video = self.load_image(path)
                else:
                    video = self.load_video(path)

                if self.use_split:
                    video = self.extract_frames(video, s=self.split_stride, f=self.num_frames)

                if self.is_i2v:
                    video, first_frame = video
                    data = {"text": text, "video": video, "path": path, "first_frame": first_frame}
                else:
                    data = {"text": text, "video": video, "path": path}
                break
            except:
                print(f"Error loading {path}, retrying...")
                data_id += 1
        return data
    

    def __len__(self):
        return len(self.path)
    

    def extract_frames(self, video, s, f):
        C, T, H, W = video.shape
        videos = []

        for start in range(0, T - f + 1, s):
            frames = video[:, start:start + f, :, :]
            videos.append(frames.unsqueeze(0))
        video = torch.stack(videos, dim=0)
        return video



class LightningModelForDataProcess(pl.LightningModule):
    def __init__(self, text_encoder_path, vae_path, num_frames, height, width, image_encoder_path=None, tiled=False, tile_size=(34, 34), tile_stride=(18, 16), use_reverse=False, use_split=False, split_stride=None):
        super().__init__()
        model_path = [text_encoder_path, vae_path]
        if image_encoder_path is not None:
            model_path.append(image_encoder_path)
        model_manager = ModelManager(torch_dtype=torch.bfloat16, device="cpu")
        model_manager.load_models(model_path)
        self.pipe = WanVideoReCamMasterPipeline.from_model_manager(model_manager)

        self.tiler_kwargs = {"tiled": tiled, "tile_size": tile_size, "tile_stride": tile_stride}
        self.data_prefix = f"{num_frames}x{height}x{width}_" if not (num_frames == 81 and height == 480 and width == 832) else ""
        if use_reverse:
            self.data_prefix += "reverse_"
        self.use_split = use_split
        self.split_stride = split_stride
        self.num_frames = num_frames


    def test_step(self, batch, batch_idx):
        try:
            text, video, path = batch["text"][0], batch["video"], batch["path"][0]
            self.pipe.device = self.device

            if self.use_split:
                videos = video[0]  # [N 1 C F H W]
                print(videos.shape)
                if len(videos) == 1:  # video 1개면 의미없음.
                    return
            else:
                videos = [video]
            for video_idx, video in enumerate(videos):
                if video is not None:
                    if self.use_split:  # path: mp4 => folder
                        path = os.path.splitext(path)[0]
                        os.makedirs(path, exist_ok=True)
                        pth_path = opj(path, f"tensor{video_idx:04d}_start{video_idx*self.split_stride:04d}.{self.data_prefix}tensors.pth")
                    else:
                        pth_path = path + f".{self.data_prefix}tensors.pth"
                    if not os.path.exists(pth_path):
                        # prompt
                        prompt_emb = self.pipe.encode_prompt(text)
                        # video
                        video = video.to(dtype=self.pipe.torch_dtype, device=self.pipe.device)
                        latents = self.pipe.encode_video(video, **self.tiler_kwargs)[0]
                        # image
                        if "first_frame" in batch:
                            first_frame = Image.fromarray(batch["first_frame"][0].cpu().numpy())
                            _, _, num_frames, height, width = video.shape
                            image_emb = self.pipe.encode_image(first_frame, num_frames, height, width)
                        else:
                            image_emb = {}
                        data = {"latents": latents, "prompt_emb": prompt_emb, "image_emb": image_emb, "num_videos": len(videos)}
                        torch.save(data, pth_path)
                    else:
                        print(f"File {pth_path} already exists, skipping.")
        except Exception as e:
            print(f"Error processing {path}: {e}")
            # 에러 경로를 파일에 기록
            with open("./error_path.txt", "a") as f:
                f.write(f"{path}\n")
            return None


class Camera(object):
    def __init__(self, c2w):
        c2w_mat = np.array(c2w).reshape(4, 4)
        self.c2w_mat = c2w_mat
        self.w2c_mat = np.linalg.inv(c2w_mat)


def check_exists(path):
    return os.path.exists(path), path


class TensorDataset(torch.utils.data.Dataset):
    def __init__(self, base_path, metadata_path, steps_per_epoch, num_frames, height, width, use_reverse=False, use_interval_pose=False, tgt_cond_window=None):
        metadata = pd.read_csv(metadata_path)
        if "MultiCamVideo-Dataset" in metadata_path:
            self.path = [os.path.join(base_path, "train", file_name) for file_name in metadata["file_name"]]
            rank_zero_info(f"{len(self.path)} videos in metadata.")

            data_prefix = f"{num_frames}x{height}x{width}_" if not (num_frames == 81 and height == 480 and width == 832) else ""
            if use_reverse: 
                data_prefix += "reverse_"
            self.path = [i + f".{data_prefix}tensors.pth" for i in tqdm(self.path, desc="checking tensor files")] # if os.path.exists(i + f".{data_prefix}tensors.pth")  # 어차피 다 있으니까 생략.
            rank_zero_info(f"{len(self.path)} tensors cached in metadata.")
        elif "DL3DV" in metadata_path:  # 각 비디오마다 몇개의 seg가 있는지 모르므로 csv를 다르게 저장해야함.
            self.path = [os.path.join(base_path, "DL3DV-ALL-960P_unzip_eachK", file_name) for file_name in metadata["file_name"]]
            rank_zero_info(f"{len(self.path)} tensors cached in metadata.")
        elif "Ego4D" in metadata_path:
            self.path = [os.path.join(base_path, "takes", file_name) for file_name in metadata["file_name"]]
            rank_zero_info(f"{len(self.path)} tensors cached in metadata.")
        else:
            assert False, f"Unknown dataset: {metadata_path}"

        assert len(self.path) > 0

        self.steps_per_epoch = steps_per_epoch
        self.num_frames = num_frames
        self.use_reverse = use_reverse
        self.use_interval_pose = use_interval_pose
        self.metadata_path = metadata_path
        self.tgt_cond_window = tgt_cond_window


    def parse_matrix(self, matrix_str):
        rows = matrix_str.strip().split('] [')
        matrix = []
        for row in rows:
            row = row.replace('[', '').replace(']', '')
            matrix.append(list(map(float, row.split())))
        return np.array(matrix)


    def get_relative_pose(self, cam_params):
        abs_w2cs = [cam_param.w2c_mat for cam_param in cam_params]
        abs_c2ws = [cam_param.c2w_mat for cam_param in cam_params]
        target_cam_c2w = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        abs2rel = target_cam_c2w @ abs_w2cs[0]
        ret_poses = [target_cam_c2w, ] + [abs2rel @ abs_c2w for abs_c2w in abs_c2ws[1:]]
        ret_poses = np.array(ret_poses, dtype=np.float32)
        return ret_poses


    def __getitem__(self, index):
        # Return: 
        # data['latents']: torch.Size([16, 21*2, 60, 104])
        # data['camera']: torch.Size([21, 3, 4])
        # data['prompt_emb']["context"][0]: torch.Size([512, 4096])
                
        while True:
            try:
                data = {}
                data_id = torch.randint(0, len(self.path), (1,))[0]
                data_id = (data_id + index) % len(self.path) # For fixed seed.
                path_tgt = self.path[data_id]

                # load the condition latent
                if "MultiCamVideo-Dataset" in self.metadata_path:
                    match = re.search(r'cam(\d+)', path_tgt)
                    tgt_idx = int(match.group(1))
                    cond_idx = random.randint(1, 10)
                    while cond_idx == tgt_idx:
                        cond_idx = random.randint(1, 10)
                    path_cond = re.sub(r'cam(\d+)', f'cam{cond_idx:02}', path_tgt)
                elif "DL3DV" in self.metadata_path:
                    stride = int(re.search(r'stride(\d+)', self.metadata_path).group(1))
                    start_idx = int(re.search(r'start(\d+)', path_tgt).group(1))
                    tgt_idx = start_idx // stride + 1
                    cond_ps = sorted(glob(opj(os.path.dirname(path_tgt), "*.pth")))
                    if self.tgt_cond_window is not None:
                        cond_idx = random.randint(max(1, tgt_idx - self.tgt_cond_window), min(len(cond_ps), tgt_idx + self.tgt_cond_window))
                    else:
                        cond_idx = random.randint(1, len(cond_ps))
                        while cond_idx == tgt_idx:
                            cond_idx = random.randint(1, len(cond_ps))
                    path_cond = cond_ps[cond_idx-1]
                elif "Ego4D" in self.metadata_path:
                    if "gp" in path_tgt:
                        prefix = "gp"
                    else:
                        prefix = "cam"

                    dir_tgt = os.path.dirname(path_tgt)

                    folder_name = path_tgt.split("/")[4]
                    cam_path = opj("./DATA/Ego4D/annotations/ego_pose/camera_pose", f"{folder_name}.json")
                    with open(cam_path, "r") as f:
                        cam_data = json.load(f)

                    indices = []
                    for k in cam_data.keys():
                        if k.startswith(prefix):
                            indices.append(int(k[-2:]))
                    if len(indices) < 2: 
                        raise ValueError(f"Indices: {indices}")
                    
                    tgt_idx = int(os.path.basename(dir_tgt)[-2:])
                    
                    cond_idx = random.sample(indices, 1)[0]
                    while cond_idx == tgt_idx:
                        cond_idx = random.sample(indices, 1)[0]
                    path_cond = path_tgt.replace(f"{prefix}{tgt_idx:02d}", f"{prefix}{cond_idx:02d}")
                else:
                    assert False, f"Unknown dataset: {self.metadata_path}"
                data_tgt = torch.load(path_tgt, weights_only=True, map_location="cpu")
                data_cond = torch.load(path_cond, weights_only=True, map_location="cpu")
                data['latents'] = torch.cat((data_tgt['latents'],data_cond['latents']),dim=1)
                data['prompt_emb'] = data_tgt['prompt_emb']
                data['image_emb'] = {}

                # load the target trajectory
                if "MultiCamVideo-Dataset" in self.metadata_path:
                    base_path = path_tgt.rsplit('/', 2)[0]
                    if "MultiCamVideo-Dataset-hy" in self.metadata_path:
                        tgt_camera_path = os.path.join(base_path, "cameras", "camera_extrinsic.json")
                    else:
                        tgt_camera_path = os.path.join(base_path, "cameras", "camera_extrinsics.json")
                    with open(tgt_camera_path, 'r') as file:
                        cam_data = json.load(file)
                elif "DL3DV" in self.metadata_path:
                    base_path = path_tgt.rsplit('/', 2)[0]
                    cam_path = opj(base_path, "transforms.json")
                    with open(cam_path, "r" ) as f:
                        cam_data = json.load(f)["frames"]

                    def transform_json(cam_data, w=41, s=24):
                        new_cam_data = {}
                        for i in range(w):
                            new_cam_data[f"frame{i}"] = {}

                        for i in range(0, len(cam_data) - w + 1, s):
                            for j in range(w):
                                new_cam_data[f"frame{j}"][f"cam{i//s + 1:02d}"] = ' '.join('[' + ' '.join(f'{x:.6g}' for x in row) + ']' for row in np.array(cam_data[i + j]["transform_matrix"]).transpose(1,0)) + " "
                                # new_cam_data[f"frame{j}"][f"cam{i//s:02d}"] = ', '.join(str(list(row)) for row in np.array(cam_data[i + j]["transform_matrix"]).transpose(1,0))
                        return new_cam_data
                    
                    cam_data = transform_json(cam_data, s=stride)
                elif "Ego4D" in self.metadata_path:
                    folder_name = path_tgt.split("/")[4]
                    cam_path = opj("./DATA/Ego4D/annotations/ego_pose/camera_pose", f"{folder_name}.json")
                    with open(cam_path, "r") as f:
                        cam_data = json.load(f)
                    
                    def transform_json(cam_data, n_frames):
                        new_cam_data = {}
                        for i in range(n_frames):
                            new_cam_data[f"frame{i}"] = {}

                        for k, v in cam_data.items():
                            if k=="metadata":
                                continue

                            k = k.replace("gp", "cam")
                            extrinsics = np.array(v["camera_extrinsics"] + [[0.0, 0.0, 0.0, 1.0]]).T.tolist()

                            row_strings = []
                            for row in extrinsics:
                                formatted = " ".join(f"{x:.6f}" for x in row)
                                row_strings.append(f"[{formatted}]")
                            extrinsics_str = " ".join(row_strings) + " "
                            
                            for i in range(n_frames):
                                new_cam_data[f"frame{i}"][f"{k}"] = extrinsics_str
                                
                        return new_cam_data
                    cam_data = transform_json(cam_data, self.num_frames)
                else:
                    assert False, f"Unknown dataset: {self.metadata_path}"
                multiview_c2ws = []

                cam_idx = list(range(self.num_frames))
                if self.use_reverse: 
                    cam_idx = cam_idx[::-1]
                cam_idx = cam_idx[::4]

                for view_idx in [cond_idx, tgt_idx]:
                    traj = [self.parse_matrix(cam_data[f"frame{idx}"][f"cam{view_idx:02d}"]) for idx in cam_idx]
                    traj = np.stack(traj).transpose(0, 2, 1)
                    c2ws = []
                    for c2w in traj:
                        if "MultiCamVideo-Dataset" in self.metadata_path:
                            c2w = c2w[:, [1, 2, 0, 3]]
                            c2w[:3, 1] *= -1.
                            c2w[:3, 3] /= 100
                        elif "Ego4D" in self.metadata_path:
                            w2c = c2w
                            c2w = np.linalg.inv(w2c)
                        c2ws.append(c2w)
                    multiview_c2ws.append(c2ws)
                cond_cam_params = [Camera(cam_param) for cam_param in multiview_c2ws[0]]
                tgt_cam_params = [Camera(cam_param) for cam_param in multiview_c2ws[1]]
                relative_poses = []
                for i in range(len(tgt_cam_params)):
                    if self.use_interval_pose:
                        if i == 0:
                            relative_pose = self.get_relative_pose([cond_cam_params[0], tgt_cam_params[i]])
                        else:
                            relative_pose = self.get_relative_pose([tgt_cam_params[i-1], tgt_cam_params[i]])
                    else:
                        relative_pose = self.get_relative_pose([cond_cam_params[0], tgt_cam_params[i]])
                    relative_poses.append(torch.as_tensor(relative_pose)[:,:3,:][1])
                pose_embedding = torch.stack(relative_poses, dim=0)  # 21x3x4
                pose_embedding = rearrange(pose_embedding, 'b c d -> b (c d)')
                data['camera'] = pose_embedding.to(torch.bfloat16)
                break
            except Exception as e:
                print(f"ERROR WHEN LOADING: {e}")
                index = random.randrange(len(self.path))
        return data
    

    def __len__(self):
        return len(self.path)


class LightningModelForTrain(pl.LightningModule):
    def __init__(
        self,
        args,
        dit_path,
        learning_rate=1e-5,
        use_gradient_checkpointing=True, use_gradient_checkpointing_offload=False,
        resume_ckpt_path=None
    ):
        super().__init__()
        model_manager = ModelManager(torch_dtype=torch.bfloat16, device="cpu")
        if os.path.isfile(dit_path):
            model_manager.load_models([dit_path])
        else:
            dit_path = dit_path.split(",")
            model_manager.load_models([dit_path])
        
        self.pipe = WanVideoReCamMasterPipeline.from_model_manager(model_manager)
        self.pipe.scheduler.set_timesteps(1000, training=True)
 
        dim=self.pipe.dit.blocks[0].self_attn.q.weight.shape[0]
        for block in self.pipe.dit.blocks:
            block.cam_encoder = nn.Linear(12, dim)
            block.cam_encoder.weight.data.zero_()
            block.cam_encoder.bias.data.zero_()
            if args.no_projector:
                pass
            else:
                block.projector = nn.Linear(dim, dim)
                block.projector.weight = nn.Parameter(torch.eye(dim))
                block.projector.bias = nn.Parameter(torch.zeros(dim))
        
        if resume_ckpt_path is not None:
            state_dict = torch.load(resume_ckpt_path, map_location="cpu")
            self.pipe.dit.load_state_dict(state_dict, strict=True)
            rank_zero_only(f"Resuming from checkpoint: {resume_ckpt_path}")

        self.freeze_parameters()
        for name, module in self.pipe.denoising_model().named_modules():
            if any(keyword in name for keyword in ["cam_encoder", "projector", "self_attn"]):
                for param in module.parameters():
                    param.requires_grad = True
        
        #### jho added >>>>
        self.pipe.dit.set_args(repeat_time_rope=args.repeat_time_rope)
        if args.use_src_adapter:
            for block in self.pipe.dit.blocks:
                block.self_attn.set_src_adapter(args.src_adapter_rank)
        for k, v in self.pipe.dit.named_parameters():
            if v.requires_grad:
                rank_zero_info(f"Trainable parameter: {k}")
        #### jho added <<<<
        
        trainable_params = 0
        seen_params = set()
        for name, module in self.pipe.denoising_model().named_modules():
            for param in module.parameters():
                if param.requires_grad and param not in seen_params:
                    trainable_params += param.numel()
                    seen_params.add(param)
        rank_zero_info(f"Total number of trainable parameters: {trainable_params}")
        
        self.learning_rate = learning_rate
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload

        self.loss_avg_meter = AverageMeter()

        
    def freeze_parameters(self):
        # Freeze parameters
        self.pipe.requires_grad_(False)
        self.pipe.eval()
        self.pipe.denoising_model().train()
        

    def training_step(self, batch, batch_idx):
        # Data
        latents = batch["latents"].to(self.device)
        prompt_emb = batch["prompt_emb"]
        prompt_emb["context"] = prompt_emb["context"][0].to(self.device)
        image_emb = batch["image_emb"]
        
        if "clip_feature" in image_emb:
            image_emb["clip_feature"] = image_emb["clip_feature"][0].to(self.device)
        if "y" in image_emb:
            image_emb["y"] = image_emb["y"][0].to(self.device)
             
        cam_emb = batch["camera"].to(self.device)

        # Loss
        self.pipe.device = self.device
        noise = torch.randn_like(latents)  # [B C 2F H W]
        timestep_id = torch.randint(0, self.pipe.scheduler.num_train_timesteps, (1,))
        timestep = self.pipe.scheduler.timesteps[timestep_id].to(dtype=self.pipe.torch_dtype, device=self.pipe.device)
        extra_input = self.pipe.prepare_extra_input(latents)
        origin_latents = copy.deepcopy(latents)  # RGB
        noisy_latents = self.pipe.scheduler.add_noise(latents, noise, timestep)
        tgt_latent_len = noisy_latents.shape[2] // 2
        noisy_latents[:, :, tgt_latent_len:, ...] = origin_latents[:, :, tgt_latent_len:, ...]
        training_target = self.pipe.scheduler.training_target(latents, noise, timestep)
        
        # Compute loss
        noise_pred = self.pipe.denoising_model()(
            noisy_latents, timestep=timestep, cam_emb=cam_emb, **prompt_emb, **extra_input, **image_emb,
            use_gradient_checkpointing=self.use_gradient_checkpointing,
            use_gradient_checkpointing_offload=self.use_gradient_checkpointing_offload
        )
        loss = torch.nn.functional.mse_loss(noise_pred[:, :, :tgt_latent_len, ...].float(), training_target[:, :, :tgt_latent_len, ...].float())
        loss = loss * self.pipe.scheduler.training_weight(timestep)

        # Record log
        self.loss_avg_meter.update(loss.item())
        self.log("train_loss_avg", self.loss_avg_meter.avg, prog_bar=True)
        return loss


    def configure_optimizers(self):
        trainable_modules = filter(lambda p: p.requires_grad, self.pipe.denoising_model().parameters())
        optimizer = torch.optim.AdamW(trainable_modules, lr=self.learning_rate)
        return optimizer
    
    @rank_zero_only
    def save_model(self, state_dict, save_path):
        torch.save(state_dict, save_path)
        
    @rank_zero_only
    def on_save_checkpoint(self, checkpoint):
        checkpoint_dir = self.trainer.checkpoint_callback.dirpath
        os.makedirs(checkpoint_dir, exist_ok=True)
        rank_zero_info(f"Checkpoint directory: {checkpoint_dir}")
        current_step = self.global_step
        rank_zero_info(f"Current step: {current_step}")

        checkpoint.clear()
        trainable_param_names = list(filter(lambda named_param: named_param[1].requires_grad, self.pipe.denoising_model().named_parameters()))
        trainable_param_names = set([named_param[0] for named_param in trainable_param_names])
        state_dict = self.pipe.denoising_model().state_dict()

        self.save_model(state_dict, os.path.join(checkpoint_dir, f"step{current_step}.ckpt"))


def parse_args():
    parser = argparse.ArgumentParser()
    
    #### jho added >>>>
    ## data
    parser.add_argument("--data_process_n_proc", type=int, default=1)
    parser.add_argument("--data_process_proc_idx", type=int, default=0)
    parser.add_argument("--data_process_use_split_video", action="store_true")
    parser.add_argument("--data_process_split_stride", type=int, default=8)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--use_reverse", action="store_true")
    parser.add_argument("--use_interval_pose", action="store_true")
    parser.add_argument("--tgt_cond_window", type=int, default=None)
    parser.add_argument("--dataset_path2", type=str, default=None)
    parser.add_argument("--metadata_file_name2", type=str, default=None)
    
    ## model
    parser.add_argument("--repeat_time_rope", action="store_true")
    parser.add_argument("--no_projector", action="store_true")
    parser.add_argument("--use_src_adapter", action="store_true")
    parser.add_argument("--src_adapter_rank", type=int, default=32)

    ## save & load
    parser.add_argument("--save_every_n_steps", type=int, default=5000)
    #### jho added <<<<

    parser.add_argument("--task", type=str, default="data_process")
    parser.add_argument("--dataset_path", type=str, default="./DATA/MultiCamVideo-Dataset")
    parser.add_argument("--output_path", type=str, default="./logs/dummy")
    parser.add_argument("--text_encoder_path", type=str, default=None)
    parser.add_argument("--image_encoder_path", type=str, default=None)
    parser.add_argument("--vae_path", type=str, default=None)
    parser.add_argument("--dit_path", type=str, default="models/Wan-AI/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors")
    parser.add_argument("--tiled", default=False, action="store_true")
    parser.add_argument("--tile_size_height", type=int, default=34)
    parser.add_argument("--tile_size_width", type=int, default=34)
    parser.add_argument("--tile_stride_height", type=int, default=18)
    parser.add_argument("--tile_stride_width", type=int, default=16)
    parser.add_argument("--steps_per_epoch", type=int, default=136000)
    parser.add_argument("--num_frames", type=int, default=81)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--dataloader_num_workers", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--accumulate_grad_batches", type=int, default=1)
    parser.add_argument("--max_epochs", type=int, default=10000)
    parser.add_argument("--training_strategy", type=str, default="deepspeed_stage_1")
    parser.add_argument("--use_gradient_checkpointing", action="store_true")
    parser.add_argument("--use_gradient_checkpointing_offload", default=False, action="store_true")
    parser.add_argument("--use_swanlab", default=False, action="store_true")
    parser.add_argument("--swanlab_mode", default=None)
    parser.add_argument("--metadata_file_name", type=str, default="metadata.csv")
    parser.add_argument("--resume_ckpt_path", type=str, default=None)
    args = parser.parse_args()
    return args


def data_process(args):
    dataset = TextVideoDataset(
        args.dataset_path,
        opj(args.dataset_path, args.metadata_file_name),
        max_num_frames=args.num_frames,
        frame_interval=1,
        num_frames=args.num_frames,
        height=args.height,
        width=args.width,
        is_i2v=args.image_encoder_path is not None,
        n_proc=args.data_process_n_proc,
        proc_idx=args.data_process_proc_idx,
        use_reverse=args.use_reverse,
        use_split=args.data_process_use_split_video,
        split_stride=args.data_process_split_stride,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=False,
        batch_size=1,
        num_workers=args.dataloader_num_workers,
    )
    model = LightningModelForDataProcess(
        text_encoder_path=args.text_encoder_path,
        image_encoder_path=args.image_encoder_path,
        vae_path=args.vae_path,
        num_frames=args.num_frames, height=args.height, width=args.width,
        tiled=args.tiled,
        tile_size=(args.tile_size_height, args.tile_size_width),
        tile_stride=(args.tile_stride_height, args.tile_stride_width),
        use_reverse=args.use_reverse,
        use_split=args.data_process_use_split_video,
        split_stride=args.data_process_split_stride,
    )
    trainer = pl.Trainer(
        accelerator="gpu",
        devices="auto",
        default_root_dir=args.output_path,
    )
    trainer.test(model, dataloader)
    
    
def train(args):
    dataset = TensorDataset(
        args.dataset_path,
        opj(args.dataset_path, args.metadata_file_name),
        steps_per_epoch=args.steps_per_epoch,
        num_frames=args.num_frames, height=args.height, width=args.width,
        use_reverse=args.use_reverse,
        use_interval_pose=args.use_interval_pose,
        tgt_cond_window=args.tgt_cond_window,
    )
    sampler = None
    
    if args.dataset_path2 is not None and args.metadata_file_name2 is not None:
        dataset2 = TensorDataset(
            args.dataset_path2,
            opj(args.dataset_path2, args.metadata_file_name2),
            steps_per_epoch=args.steps_per_epoch,
            num_frames=args.num_frames, height=args.height, width=args.width,
            use_reverse=args.use_reverse,
            use_interval_pose=args.use_interval_pose,
            tgt_cond_window=args.tgt_cond_window,
        )

        rank_zero_info(f"Dataset 1: {len(dataset)} samples")
        rank_zero_info(f"Dataset 2: {len(dataset2)} samples")
        rank_zero_info(f"Total samples: {len(dataset) + len(dataset2)}")
        
        len1 = len(dataset)
        len2 = len(dataset2)
        total_len = len1 + len2
        weights = np.concatenate([
            np.full(len1, 1 / len1),
            np.full(len2, 1 / len2),
        ])
        sampler = WeightedRandomSampler(weights, num_samples=total_len, replacement=True)
        dataset = ConcatDataset([dataset, dataset2])
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=True if sampler is None else False,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
        sampler=sampler,
    )
    model = LightningModelForTrain(
        args=args,
        dit_path=args.dit_path,
        learning_rate=args.learning_rate,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        resume_ckpt_path=args.resume_ckpt_path,
    )

    if args.use_swanlab:
        from swanlab.integration.pytorch_lightning import SwanLabLogger
        swanlab_config = {"UPPERFRAMEWORK": "DiffSynth-Studio"}
        swanlab_config.update(vars(args))
        swanlab_logger = SwanLabLogger(
            project="wan",
            name="wan",
            config=swanlab_config,
            mode=args.swanlab_mode,
            logdir=os.path.join(args.output_path, "swanlog"),
        )
        logger = [swanlab_logger]
    else:
        logger = None
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="gpu",
        devices="auto",
        precision="bf16",
        strategy=args.training_strategy,
        default_root_dir=args.output_path,
        accumulate_grad_batches=args.accumulate_grad_batches,
        callbacks=[pl.pytorch.callbacks.ModelCheckpoint(save_top_k=-1, every_n_train_steps=args.save_every_n_steps)],
        logger=logger,
    )
    
    rank_zero_info("#### Training Configuration ####")
    rank_zero_info(f"Max epoch: {args.max_epochs}")
    rank_zero_info(f"Total batch size: {args.train_batch_size * args.accumulate_grad_batches * trainer.num_devices}")
    rank_zero_info("#"*30)
    
    trainer.fit(model, dataloader)
    


if __name__ == '__main__':
    args = parse_args()
    print("import & args done...")
    os.makedirs(os.path.join(args.output_path, "checkpoints"), exist_ok=True)
    rank_zero_info(f"Output path: {args.output_path}")

    with open(os.path.join(args.output_path, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    if args.task == "data_process":
        data_process(args)
    elif args.task == "train":
        train(args)
