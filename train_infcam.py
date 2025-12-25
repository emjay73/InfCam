import copy
import os
import re
import torch, os, imageio, argparse
from torchvision.transforms import v2
from einops import rearrange
import lightning as pl
import pandas as pd
from diffsynth import WanVideoInfCamPipeline, ModelManager, load_state_dict
import torchvision
from PIL import Image
import numpy as np
import random
import json
import torch.nn as nn
import torch.nn.functional as F
import shutil
from torch.utils.data import ConcatDataset

from diffsynth.models.wan_video_dit import SelfAttentionPre
import random
from lightning.pytorch.loggers import WandbLogger
import glob
import math
from torch.utils.data import ConcatDataset
from torch.utils.data import WeightedRandomSampler

from tqdm import tqdm
from pytorch_lightning.utilities import rank_zero_info, rank_zero_only

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class CodeUploadCallback(pl.Callback):
    """
    On training start, upload all .py files to WandB.
    """
    def on_train_start(self, trainer, pl_module):
        # Only run on the main process
        if not trainer.is_global_zero:
            return

        logger = trainer.logger
        if isinstance(logger, WandbLogger):
            # The modern way to log code with wandb, respects .gitignore
            logger.experiment.log_code(".")
            print("WandB: Uploading project code...")

def make_pth_path( path, height, width, num_frames ):
    if height == 480 and width == 832:
        suffix = ".tensors.pth"        
    else:        
        suffix = f".{num_frames}x{height}x{width}_tensors.pth"
    pth_path = path + suffix

    return pth_path

# data process를 위한 class
class TextVideoDataset(torch.utils.data.Dataset):
    def __init__(self, base_path, metadata_path, max_num_frames=81, frame_interval=1, num_frames=81, height=480, width=832, is_i2v=False,          
    ):
        metadata = pd.read_csv(metadata_path)
        self.path = [os.path.join(base_path, "train", file_name) for file_name in metadata["file_name"]]        
        self.text = metadata["text"].to_list()
        
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
        
        self.valid_ids = []
        for i, p in tqdm(enumerate(self.path), desc="mapping valid ids .."):
            pth = make_pth_path(p, self.height, self.width, self.num_frames)
            if not os.path.exists(pth):
                self.valid_ids.append(i)        
        
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
        if reader.count_frames() < max_num_frames or reader.count_frames() - 1 < start_frame_id + (num_frames - 1) * interval:
            reader.close()
            return None
        
        frames = []
        first_frame = None
        for frame_id in range(num_frames):
            frame = reader.get_data(start_frame_id + frame_id * interval)
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

    def load_video(self, file_path, start_frame_id=0):  
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


    def __getitem__(self, idx):
        while True:
            data_id = self.valid_ids[idx]
            text = self.text[data_id]
            path = self.path[data_id]
            
            try:
                if self.is_image(path):
                    if self.is_i2v:
                        raise ValueError(f"{path} is not a video. I2V model doesn't support image-to-image training.")
                    video = self.load_image(path)
                else:

                    video = self.load_video(path)
                    
                if self.is_i2v:
                    video, first_frame = video
                    data = {"text": text, "video": video, "path": path, "first_frame": first_frame}
                else:
                    data = {"text": text, "video": video, "path": path}
                break
            except:
                data_id += 1
        return data
    


    def __len__(self):        
        return len(self.valid_ids)       

class LightningModelForDataProcess(pl.LightningModule):
    
    def __init__(self, text_encoder_path, vae_path, image_encoder_path=None, tiled=False, tile_size=(34, 34), tile_stride=(18, 16), 
                    num_frames=81, height=480, width=832): #, start_frame_id = 0):
    
        super().__init__()
        model_path = [text_encoder_path, vae_path]
        if image_encoder_path is not None:
            model_path.append(image_encoder_path)
        model_manager = ModelManager(torch_dtype=torch.bfloat16, device="cpu")
        model_manager.load_models(model_path)
        self.pipe = WanVideoInfCamPipeline.from_model_manager(model_manager)        
        self.height = height
        self.width = width
        
        self.tiler_kwargs = {"tiled": tiled, "tile_size": tile_size, "tile_stride": tile_stride}
        
    def test_step(self, batch, batch_idx):
        text, video, path = batch["text"][0], batch["video"], batch["path"][0]
        
        self.pipe.device = self.device
        if video is not None:            
            pth_path = make_pth_path(path, self.height, self.width, self.num_frames)
            
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
                data = {"latents": latents, "prompt_emb": prompt_emb, "image_emb": image_emb}
                torch.save(data, pth_path)
            else:
                print(f"File {pth_path} already exists, skipping.")

class Camera(object):
    def __init__(self, c2w):
        c2w_mat = np.array(c2w).reshape(4, 4)
        self.c2w_mat = c2w_mat
        self.w2c_mat = np.linalg.inv(c2w_mat)


# data loading을 위한 class
class TensorDataset(torch.utils.data.Dataset):
    
    def __init__(self, base_path, metadata_path, steps_per_epoch, num_frames=81, height=480, width=832, use_multicam_reverse=False):
        
        self.num_frames = num_frames
        self.height = height
        self.width = width
       
        metadata = pd.read_csv(metadata_path)
        if "AugMCV" in metadata_path or "MultiCamVideo-Dataset" in metadata_path or "SynCamVideo-Dataset" in metadata_path:
            if 'test' in metadata_path:
                self.path = [os.path.join(base_path, "test", file_name) for file_name in metadata["file_name"]]
            else:                
                self.path = [os.path.join(base_path, "train", file_name) for file_name in metadata["file_name"] if "_aug" not in file_name]
                self.vidpath2aug = {}
                for file_name in tqdm(metadata["file_name"], desc="mapping aug data .."):
                    if "_aug" not in file_name:
                        continue
                    aug_path = os.path.join(base_path, "train", file_name)
                    setting = "_".join(file_name.split("/")[0].split("_")[:-1])
                    scene = "_".join(file_name.split("/")[1].split("_")[:-1])
                    file_name_only = file_name.split("/")[-1]
                    origin = os.path.join(setting, scene, 'videos',file_name_only)
                    self.vidpath2aug[origin] = file_name
                for file_name in tqdm(metadata["file_name"], desc="mapping aug data .."):
                    if "_aug" in file_name:
                        continue
                    if file_name in self.vidpath2aug:
                        continue
                    self.vidpath2aug[file_name] = None                      
                
            rank_zero_info(f"{len(self.path)} videos in metadata.")
            tmp_path = []
            if num_frames == 81 and height == 480 and width == 832:
                suffix = ".tensors.pth"
                for i in tqdm(self.path, desc="Searching for tensor files"):
                    tmp_path.append(i + suffix)
                    
                self.path = tmp_path
            else:
                for i in tqdm(self.path, desc="Searching for tensor files"):
                    suffix = f".{num_frames}x{height}x{width}_tensors.pth"
                    tmp_path.append(i + suffix)

                    if use_multicam_reverse and "MultiCamVideo-Dataset" in metadata_path:
                        suffix = f"{num_frames}x{height}x{width}_reverse_tensors.pth"
                        tmp_path.append(i + suffix)
                self.path = tmp_path
        elif "DL3DV" in metadata_path:  # 각 비디오마다 몇개의 seg가 있는지 모르므로 csv를 다르게 저장해야함.
            self.path = [os.path.join(base_path, "DL3DV-ALL-960P_unzip_eachK", file_name) for file_name in metadata["file_name"]]
        elif "Ego4D" in metadata_path:
            self.path = [os.path.join(base_path, "takes", file_name) for file_name in metadata["file_name"]]
        else:
            assert False, f"Unknown dataset: {metadata_path}"
        rank_zero_info(f"{len(self.path)} tensors cached in metadata.")

        assert len(self.path) > 0
        self.steps_per_epoch = steps_per_epoch
        self.metadata_path = metadata_path

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
        return ret_poses # target to source(init frame)

    
    def path2intrinsic(self, path_video):
        base_path = path_video.rsplit('/', 2)[0] # ./DATA/AugMCV/train/f24_aperture5_aug/scene1823_1_f50
        _, cam_setting, scene = base_path.rsplit("/", 2)

        if "aug" not in cam_setting:
            match = re.search(r'f(\d+)', cam_setting)
        else:
            match = re.search(r'f(\d+)', scene)
        if match:
            focal_mm =  int(match.group(1))
        
        # recam & syncam sensor config 
        # resolution: 1280x1280, sensor size: 23.76mm x 23.76mm
        focal_px = focal_mm * max(self.height, self.width) / 23.76 
        intrinsic = torch.tensor([focal_px, focal_px, self.width//2, self.height//2]).to(torch.bfloat16)
        return intrinsic
    
    def aug_tensor_path(self, path_tensor):
        path_vid_rel, tensor_name, ext = '/'.join(path_tensor.split('/')[-4:]).rsplit('.',2)
        if (self.vidpath2aug[path_vid_rel] is not None) :            
            path_vid_aug = os.path.join( path_tensor.rsplit('/', 4)[0], self.vidpath2aug[path_vid_rel] )
            path_tensor_aug = path_vid_aug + '.' + tensor_name + '.' + ext
            return path_tensor_aug
        else:
            return None
    

    def __getitem__(self, index):

        while True:            
            try:                
                data = {}
                data_id = torch.randint(0, len(self.path), (1,))[0]
                data_id = (data_id + index) % len(self.path) # For fixed seed.
                path_tgt = self.path[data_id]
                
                path_aug = self.aug_tensor_path(path_tgt)
                if (path_aug is not None) and (random.random() < 0.5):
                    if (random.random() < 0.5):
                        path_cond = path_aug
                        path_tgt = path_tgt
                    else:
                        path_cond = path_tgt
                        path_tgt = path_aug
                else:
                    path_cond = path_tgt                    
                
                data_tgt = torch.load(path_tgt, weights_only=True, map_location="cpu")

                # load the condition latent
                match = re.search(r'cam(\d+)', path_tgt)
                tgt_idx = int(match.group(1))
                
                cond_idx = random.randint(1, 10)
                while cond_idx == tgt_idx:
                    cond_idx = random.randint(1, 10)

                
                path_cond = re.sub(r'cam(\d+)', f'cam{cond_idx:02}', path_cond)
                data_cond = torch.load(path_cond, weights_only=True, map_location="cpu")
                
                data['latents'] = torch.cat((data_tgt['latents'],data_cond['latents']),dim=1) # frame
                data['prompt_emb'] = data_tgt['prompt_emb']
                data['image_emb'] = {}                          
                data['intrinsic_trg'] = self.path2intrinsic(path_tgt)
                data['intrinsic_cond'] = self.path2intrinsic(path_cond)
                
                # load the target trajectory
                base_path = path_tgt.rsplit('/', 2)[0]

                tgt_camera_path = os.path.join(base_path, "cameras", "camera_extrinsics.json")
                with open(tgt_camera_path, 'r') as file:
                    cam_data_trg = json.load(file)
                
                base_path = path_cond.rsplit('/', 2)[0]                
                cond_camera_path = os.path.join(base_path, "cameras", "camera_extrinsics.json")
                with open(cond_camera_path, 'r') as file:
                    cam_data_cond = json.load(file)
                
                multiview_c2ws = []            
                
                cam_idx = list(range(self.num_frames))[::4] 
                cam_datas = [cam_data_cond, cam_data_trg]
                
                for view_idx, cam_data in zip( [cond_idx, tgt_idx], cam_datas):
                    traj = [self.parse_matrix(cam_data[f"frame{idx}"][f"cam{view_idx:02d}"]) for idx in cam_idx]
                    traj = np.stack(traj).transpose(0, 2, 1)
                    c2ws = []
                    for c2w in traj:
                        c2w = c2w[:, [1, 2, 0, 3]]
                        c2w[:3, 1] *= -1.
                        c2w[:3, 3] /= 100
                        c2ws.append(c2w)
                    multiview_c2ws.append(c2ws)
                cond_cam_params = [Camera(cam_param) for cam_param in multiview_c2ws[0]]
                tgt_cam_params = [Camera(cam_param) for cam_param in multiview_c2ws[1]]
                relative_poses = []
                
                for i in range(len(tgt_cam_params)):                                 
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

    def get_random_palindrome_slice(self, initial_list):
        """
        리스트를 뒤집어 연장한 후, 임의의 위치에서 원래 길이만큼의 부분을 추출합니다.
        """
        # 1. 입력 리스트가 비어있으면 빈 리스트를 반환.
        if not initial_list:
            return []

        n = len(initial_list)

        # 2. 리스트를 뒤집어서 연장. (예: [1,2,3] -> [1,2,3,2,1])        
        pool_list = initial_list + initial_list[-2::-1]

        # 3. 추출을 시작할 임의의 인덱스를 결정.
        #    결과 리스트의 길이가 n이어야 하므로, 시작점은 0부터 n-1까지만 가능.        
        max_start_index = n-1
        start_index = random.randint(0, max_start_index)

        # 4. 임의의 시작점부터 원래 리스트 길이(n)만큼 잘라냅니다.
        result = pool_list[start_index : start_index + n]
        
        print(start_index, start_index + n, 2*n -1)
        return result
    
class LightningModelForTrain(pl.LightningModule):
    def __init__(
        self,
        dit_path,
        learning_rate=1e-5,
        use_gradient_checkpointing=True, use_gradient_checkpointing_offload=False,
        resume_ckpt_path=None,        
    ):
        super().__init__()

        model_manager = ModelManager(torch_dtype=torch.bfloat16, device="cpu")
        if os.path.isfile(dit_path):
            model_manager.load_models([dit_path])
        else:
            dit_path = dit_path.split(",")
            model_manager.load_models([dit_path])
        
        self.pipe = WanVideoInfCamPipeline.from_model_manager(model_manager)
        self.pipe.scheduler.set_timesteps(1000, training=True)

        dim=self.pipe.dit.blocks[0].self_attn.q.weight.shape[0]

        for i, block in enumerate(self.pipe.dit.blocks):
            block.self_attn_pre = SelfAttentionPre(block.self_attn.dim, block.self_attn.num_heads)
            block.norm0 = nn.LayerNorm(block.dim, elementwise_affine=False)
            block.modulation0 = nn.Parameter(torch.randn(1,6,block.dim)/block.dim**0.5)
                
            block.self_attn_pre.load_state_dict(block.self_attn.state_dict(), strict=False)
            block.norm0.load_state_dict(block.norm1.state_dict(), strict=False)
            block.modulation0.data.copy_(block.modulation.data)
                
            block.cam_encoder = nn.Linear(16, dim)                
            block.cam_encoder.weight.data.zero_()
            block.cam_encoder.bias.data.zero_()
            
            block.zero_conv = nn.Conv3d(1536, 1536, kernel_size=(1, 1, 1), stride=(1, 1, 1))
            block.zero_conv.weight.data.zero_()
            block.zero_conv.bias.data.zero_()

        
        if resume_ckpt_path is not None:
            state_dict = torch.load(resume_ckpt_path, map_location="cpu")
            self.pipe.dit.load_state_dict(state_dict, strict=True)
            rank_zero_info(f"Loaded DIT state dict from {resume_ckpt_path}")
            rank_zero_info(f"Loaded DIT state dict from {resume_ckpt_path}")
            rank_zero_info(f"Loaded DIT state dict from {resume_ckpt_path}")
            rank_zero_info(f"Loaded DIT state dict from {resume_ckpt_path}")

        self.freeze_parameters()

        for name, module in self.pipe.denoising_model().named_modules():           
            if any(keyword in name for keyword in ["cam_encoder", "cam_kt_encoder", "projector", "self_attn_pre", "norm0", "modulation0", "zero_conv", "linear"]):                

                for name_sub, param in module.named_parameters():    
                    rank_zero_info(f"Trainable: {name}.{name_sub}")
                    param.requires_grad = True        

        trainable_params = 0
        seen_params = set()
        for name, module in self.pipe.denoising_model().named_modules():
            for param in module.parameters():
                if param.requires_grad and param not in seen_params:
                    trainable_params += param.numel()
                    seen_params.add(param)
        print(f"Total number of trainable parameters: {trainable_params}")
        
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
        latents = batch["latents"].to(self.device) # [b, c, ((f-1)/4+1)*2, h/8, w/8] = [1, 16, 21*2, 60, 104]
        prompt_emb = batch["prompt_emb"] # [b, 1, 512, 4096]
        prompt_emb["context"] = prompt_emb["context"][0].to(self.device)
        image_emb = batch["image_emb"] # {}
        
        if "clip_feature" in image_emb:
            image_emb["clip_feature"] = image_emb["clip_feature"][0].to(self.device)
        if "y" in image_emb:
            image_emb["y"] = image_emb["y"][0].to(self.device)
        
        cam_intrinsic =(batch["intrinsic_cond"].to(self.device), batch["intrinsic_trg"].to(self.device)) # [b, 4] = [1, 4]
        cam_extrinsic = batch["camera"].to(self.device) # [b, ((f-1)/4+1), 3*4] = [1, 21, 12]
       
        # Loss
        self.pipe.device = self.device
        noise = torch.randn_like(latents)
        timestep_id = torch.randint(0, self.pipe.scheduler.num_train_timesteps, (1,))
        timestep = self.pipe.scheduler.timesteps[timestep_id].to(dtype=self.pipe.torch_dtype, device=self.pipe.device)
        extra_input = self.pipe.prepare_extra_input(latents) # {}
        origin_latents = copy.deepcopy(latents)
        noisy_latents = self.pipe.scheduler.add_noise(latents, noise, timestep)
        tgt_latent_len = noisy_latents.shape[2] // 2
        noisy_latents[:, :, tgt_latent_len:, ...] = origin_latents[:, :, tgt_latent_len:, ...]
        training_target = self.pipe.scheduler.training_target(latents, noise, timestep) # noise - latents
        
        size_batch, size_channel, size_frame, size_height, size_width = noisy_latents.shape
        noisy_latents = noisy_latents.reshape(size_batch, size_channel, 2, size_frame // 2, size_height, size_width).permute(2, 0, 1, 3, 4, 5).reshape(2*size_batch, size_channel, size_frame // 2, size_height, size_width)
        training_target = training_target.reshape(size_batch, size_channel, 2, size_frame // 2, size_height, size_width).permute(2, 0, 1, 3, 4, 5).reshape(2*size_batch, size_channel, size_frame // 2, size_height, size_width)
        
        # Compute loss        
        noise_pred = self.pipe.denoising_model()(
            noisy_latents, timestep=timestep,             
            cam_intrinsic=cam_intrinsic,
            cam_extrinsic=cam_extrinsic,
            **prompt_emb, **extra_input, **image_emb,
            use_gradient_checkpointing=self.use_gradient_checkpointing,
            use_gradient_checkpointing_offload=self.use_gradient_checkpointing_offload,           
        )

        loss = torch.nn.functional.mse_loss(noise_pred[:size_batch, :, :, ...].float(), training_target[:size_batch, :, :, ...].float())
        loss = loss * self.pipe.scheduler.training_weight(timestep)

        # Record log
        self.loss_avg_meter.update(loss.item())
        self.log("train_loss_avg", self.loss_avg_meter.avg, prog_bar=True)
        return loss


    def configure_optimizers(self):
        trainable_modules = filter(lambda p: p.requires_grad, self.pipe.denoising_model().parameters())
        optimizer = torch.optim.AdamW(trainable_modules, lr=self.learning_rate)
        return optimizer
    

    def on_save_checkpoint(self, checkpoint):        
        if not self.trainer.is_global_zero:
            return
        
        checkpoint_dir = self.trainer.checkpoint_callback.dirpath

        os.makedirs(checkpoint_dir, exist_ok=True)        
        print(f"Checkpoint directory: {checkpoint_dir}")
        current_step = self.global_step
        print(f"Current step: {current_step}")

        checkpoint.clear()
        trainable_param_names = list(filter(lambda named_param: named_param[1].requires_grad, self.pipe.denoising_model().named_parameters()))
        trainable_param_names = set([named_param[0] for named_param in trainable_param_names])
        state_dict = self.pipe.denoising_model().state_dict()
        torch.save(state_dict, os.path.join(checkpoint_dir, f"step{current_step}.ckpt"))



def parse_args():
    parser = argparse.ArgumentParser(description="Train InfCam")   
    parser.add_argument("--use_multicam_reverse", action="store_true")

    parser.add_argument(
        "--task",
        type=str,
        default="data_process",
        required=True,
        choices=["data_process", "train"],
        help="Task. `data_process` or `train`.",
    )
    parser.add_argument(
        "--dataset_path",        
        nargs="+",        
        type=str,
        default=None,
        required=True,
        help="The path of the Dataset. (ex: --dataset_path ./path1 ./path2)",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./",
        help="Path to save the model.",
    )
    parser.add_argument(
        "--text_encoder_path",
        type=str,
        default=None,
        help="Path of text encoder.",
    )
    parser.add_argument(
        "--image_encoder_path",
        type=str,
        default=None,
        help="Path of image encoder.",
    )
    parser.add_argument(
        "--vae_path",
        type=str,
        default=None,
        help="Path of VAE.",
    )
    parser.add_argument(
        "--dit_path",
        type=str,
        default="models/Wan-AI/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors",
        help="Path of DiT.",
    )
    parser.add_argument(
        "--tiled",
        default=False,
        action="store_true",
        help="Whether enable tile encode in VAE. This option can reduce VRAM required.",
    )
    parser.add_argument(
        "--tile_size_height",
        type=int,
        default=34,
        help="Tile size (height) in VAE.",
    )
    parser.add_argument(
        "--tile_size_width",
        type=int,
        default=34,
        help="Tile size (width) in VAE.",
    )
    parser.add_argument(
        "--tile_stride_height",
        type=int,
        default=18,
        help="Tile stride (height) in VAE.",
    )
    parser.add_argument(
        "--tile_stride_width",
        type=int,
        default=16,
        help="Tile stride (width) in VAE.",
    )
    parser.add_argument(
        "--steps_per_epoch",
        type=int,
        default=170000,
        help="Number of steps per epoch.",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=81,
        help="Number of frames.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Image height.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=832,
        help="Image width.",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=4,
        help="Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="Learning rate.",
    )
    parser.add_argument(
        "--accumulate_grad_batches",
        type=int,
        default=1,
        help="The number of batches in gradient accumulation.",
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=100,
        help="Number of epochs.",
    )
    parser.add_argument(
        "--training_strategy",
        type=str,
        default="deepspeed_stage_1",
        choices=["auto", "deepspeed_stage_1", "deepspeed_stage_2", "deepspeed_stage_3"],
        help="Training strategy",
    )
    parser.add_argument(
        "--use_gradient_checkpointing",
        default=False,
        action="store_true",
        help="Whether to use gradient checkpointing.",
    )
    parser.add_argument(
        "--use_gradient_checkpointing_offload",
        default=False,
        action="store_true",
        help="Whether to use gradient checkpointing offload.",
    )
    parser.add_argument(
        "--use_swanlab",
        default=False,
        action="store_true",
        help="Whether to use SwanLab logger.",
    )
    parser.add_argument(
        "--swanlab_mode",
        default=None,
        help="SwanLab mode (cloud or local).",
    )
    
    parser.add_argument(
        "--metadata_file_name",
        type=str,
        default="metadata.csv",
    )
    parser.add_argument(
        "--resume_ckpt_path",
        type=str,
        default=None,
    )
    
    parser.add_argument(
        "--use_wandb",
        default=False,
        action="store_true",
        help="Whether to use WandB logger.",
    )
    parser.add_argument(        
        "--project_name",
        type=str,
        default="InfCam",
        help="WandB project name.",
    )    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=3,
    )    
    parser.add_argument(
        "--log_every_n_steps",
        type=int,
        default=50,
        help="How often to log within steps. Default is 50.",
    )
    parser.add_argument(
        "--save_every_n_steps",
        type=int,
        default=1000,
        help="How often to save a checkpoint within steps. Default is None (end of epoch).",
    )

    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Specify a name for the run, which will be used as the directory name for saving checkpoints.",
    )
    
    args = parser.parse_args()
    return args


def data_process(args):
    
    # 각 경로의 데이터셋을 담을 리스트를 생성합니다.
    dataset_list = []
    print(f"Loading datasets from: {args.dataset_path}")

    # 인자로 받은 모든 경로에 대해 반복합니다.
    for path in args.dataset_path:        
        metadata_path = os.path.join(path, args.metadata_file_name)
        if not os.path.exists(metadata_path):
            print(f"Warning: '{metadata_path}'를 찾을 수 없어 해당 경로를 건너뜁니다.")
            continue
        
        # 각 경로에 대해 개별 데이터셋 객체를 생성합니다.
        single_dataset = TextVideoDataset(
            base_path=path,
            metadata_path=metadata_path,
            max_num_frames=args.num_frames,
            frame_interval=1,
            num_frames=args.num_frames,
            height=args.height,
            width=args.width,
            is_i2v=args.image_encoder_path is not None,
            
        )
        dataset_list.append(single_dataset)

    # 데이터셋 리스트가 비어있으면 에러를 발생시킵니다.
    if not dataset_list:
        raise ValueError("유효한 데이터셋을 하나도 불러오지 못했습니다. dataset_path를 확인해주세요.")

    # ConcatDataset으로 모든 데이터셋을 하나의 큰 데이터셋으로 합칩니다.
    dataset_concat = ConcatDataset(dataset_list)
    print(f"Successfully loaded {len(dataset_list)} datasets with a total of {len(dataset_concat)} items.")
    
    dataloader = torch.utils.data.DataLoader(
        dataset_concat,
        shuffle=False,        
        batch_size=args.batch_size,        
        num_workers=args.dataloader_num_workers,
                
        persistent_workers=True,
        pin_memory=True,
        prefetch_factor=max(1, args.batch_size // args.dataloader_num_workers),        
    )
    model = LightningModelForDataProcess(
        text_encoder_path=args.text_encoder_path,
        image_encoder_path=args.image_encoder_path,
        vae_path=args.vae_path,
        tiled=args.tiled,
        tile_size=(args.tile_size_height, args.tile_size_width),
        tile_stride=(args.tile_stride_height, args.tile_stride_width),
        
        num_frames=args.num_frames,
        height=args.height,
        width=args.width,        
    )
    trainer = pl.Trainer(
        accelerator="gpu",
        devices="auto",
        default_root_dir=args.output_path,
    )

    trainer.test(model, dataloader)
    
    
def train(args):
    dataset_list = []
    print(f"Loading datasets from: {args.dataset_path}")
    
    for path in args.dataset_path:
        # metadata_path = os.path.join(path, "metadata.csv")
        metadata_path = os.path.join(path, args.metadata_file_name)
        if not os.path.exists(metadata_path):
            print(f"Warning: '{metadata_path}'를 찾을 수 없어 해당 경로를 건너뜁니다.")
            continue
                
        single_dataset = TensorDataset(
            base_path=path,
            metadata_path=metadata_path,
            steps_per_epoch=args.steps_per_epoch,
            num_frames=args.num_frames,
            height=args.height,
            width=args.width,
            use_multicam_reverse=args.use_multicam_reverse,
        )
        dataset_list.append(single_dataset)
    
    if not dataset_list:
        raise ValueError("유효한 데이터셋을 하나도 불러오지 못했습니다. dataset_path를 확인해주세요.")

    dataset_sizes = []
    for dataset in dataset_list:
        dataset_sizes.append(len(dataset))

    # Calculate weights: assign inverse weights based on the largest dataset size
    max_size = max(dataset_sizes)
    weights = []
    cumulative_size = 0

    for i, dataset_size in enumerate(dataset_sizes):
        # Assign higher weights to smaller datasets
        weight_per_sample = max_size / dataset_size
        weights.extend([weight_per_sample] * dataset_size)
        print(f"Dataset {i}: size={dataset_size}, weight_per_sample={weight_per_sample:.3f}")

    dataset_concat = ConcatDataset(dataset_list)
    print(f"Successfully loaded {len(dataset_list)} datasets with a total of {len(dataset_concat)} items.")
        
    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=len(dataset_concat),  # Number of samples per epoch
        replacement=True  # Allow sampling with replacement
    )
    
    dataloader = torch.utils.data.DataLoader(
        dataset_concat,
                
        batch_size=args.batch_size,        
        # sampler=sampler,  # if you want to use sampler, set shuffle=False
        num_workers=args.dataloader_num_workers,        
        shuffle=True,
        
        # comment out when debugging --------------
        persistent_workers=True,
        pin_memory=True,
        prefetch_factor=max(1, args.batch_size // args.dataloader_num_workers),
        # ------------------------------------
    )
    model = LightningModelForTrain(
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
    elif args.use_wandb:
        from lightning.pytorch.loggers import WandbLogger
        wandb_logger = WandbLogger(            
            project=args.project_name,        
            name = args.run_name,
            config=vars(args),
            save_dir=args.output_path,
        )
        logger = [wandb_logger]
    else:
        logger = None
    
    if args.run_name:
        output_dir = os.path.join(args.output_path, args.project_name, args.run_name)
        os.makedirs(output_dir, exist_ok=True)
    else:
        output_dir = args.output_path
    
    checkpoint_callback = pl.pytorch.callbacks.ModelCheckpoint(
        dirpath=os.path.join(output_dir, "checkpoints"),
        save_top_k=-1, # epoch 끝날 때 마다
        every_n_train_steps=args.save_every_n_steps
    )
    
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="gpu",
        devices="auto",
        precision="bf16",
        strategy=args.training_strategy,        
        default_root_dir=output_dir,        
        accumulate_grad_batches=args.accumulate_grad_batches,        
        callbacks=[checkpoint_callback, CodeUploadCallback()],        
        logger=logger,
        log_every_n_steps=args.log_every_n_steps,
    )
    trainer.fit(model, dataloader)


if __name__ == '__main__':
    args = parse_args()
    os.makedirs(os.path.join(args.output_path, "checkpoints"), exist_ok=True)
    if args.task == "data_process":
        data_process(args)
    elif args.task == "train":
        train(args)
