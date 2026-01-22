import sys
import torch
import torch.nn as nn
from diffsynth import ModelManager, WanVideoReCamMasterPipeline, save_video, VideoData
import torch, os, imageio, argparse
from torchvision.transforms import v2
from einops import rearrange
import pandas as pd
import torchvision
from PIL import Image
import numpy as np
import json
from ipdb import set_trace as breakpoint
from utils import concat_videos_hori

class Camera(object):
    def __init__(self, c2w):
        c2w_mat = np.array(c2w).reshape(4, 4)
        self.c2w_mat = c2w_mat
        self.w2c_mat = np.linalg.inv(c2w_mat)

class TextVideoCameraDataset(torch.utils.data.Dataset):
    def __init__(self, base_path, metadata_path, args, max_num_frames=81, frame_interval=1, num_frames=81, height=480, width=832, is_i2v=False, use_reverse=False, use_interval_pose=False, tgt_cam_path=None, repeat_first_frame=None):
        metadata = pd.read_csv(metadata_path)
        self.path = [os.path.join(base_path, "videos", file_name) for file_name in metadata["file_name"]]
        self.text = metadata["text"].to_list()
        
        self.max_num_frames = max_num_frames
        self.frame_interval = frame_interval
        self.num_frames = num_frames
        self.height = height
        self.width = width
        self.is_i2v = is_i2v
        self.args = args
        self.cam_type = self.args.cam_type
        self.use_reverse = use_reverse
        self.use_interval_pose = use_interval_pose
        self.tgt_cam_path = tgt_cam_path
        self.repeat_first_frame = repeat_first_frame
            
        self.frame_process = v2.Compose([
            v2.CenterCrop(size=(height, width)),
            v2.Resize(size=(height, width), antialias=True),
            v2.ToTensor(),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        
        
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


    def is_image(self, file_path):
        file_ext_name = file_path.split(".")[-1]
        if file_ext_name.lower() in ["jpg", "jpeg", "png", "webp"]:
            return True
        return False
    

    def load_video(self, file_path):
        # start_frame_id = torch.randint(0, self.max_num_frames - (self.num_frames - 1) * self.frame_interval, (1,))[0]
        start_frame_id = 0
        frames = self.load_frames_using_imageio(file_path, self.max_num_frames, start_frame_id, self.frame_interval, self.num_frames, self.frame_process)
        return frames


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

        cam_to_origin = 0
        target_cam_c2w = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, -cam_to_origin],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        abs2rel = target_cam_c2w @ abs_w2cs[0]
        ret_poses = [target_cam_c2w, ] + [abs2rel @ abs_c2w for abs_c2w in abs_c2ws[1:]]
        # ret_poses = [target_cam_c2w, ] + [abs_c2w for abs_c2w in abs_c2ws[1:]]
        ret_poses = np.array(ret_poses, dtype=np.float32)
        return ret_poses


    def __getitem__(self, data_id):
        text = self.text[data_id]
        path = self.path[data_id]
        video = self.load_video(path)
        if self.repeat_first_frame is not None:
            n_repeat_frame = int(self.repeat_first_frame)
            video = torch.cat([video[:, :1].repeat(1, n_repeat_frame, 1, 1), video], dim=1)

        if video is None:
            raise ValueError(f"{path} is not a valid video.")
        num_frames = video.shape[1]
        # assert num_frames == 81
        data = {"text": text, "video": video, "path": path}

        # load camera
        if self.tgt_cam_path is not None:
            tgt_camera_path = self.tgt_cam_path
        else:
            tgt_camera_path = "./example_test_data/cameras/camera_extrinsics.json"
        with open(tgt_camera_path, 'r') as file:
            cam_data = json.load(file)

        cam_idx = list(range(num_frames))
        if self.use_reverse:
            cam_idx = cam_idx[::-1]
        cam_idx = cam_idx[::4]
        
        traj = [self.parse_matrix(cam_data[f"frame{idx}"][f"cam{int(self.cam_type):02d}"]) for idx in cam_idx]
        traj = np.stack(traj).transpose(0, 2, 1)
        c2ws = []
        for c2w in traj:
            c2w = c2w[:, [1, 2, 0, 3]]
            c2w[:3, 1] *= -1.
            c2w[:3, 3] /= 100
            c2ws.append(c2w)
        tgt_cam_params = [Camera(cam_param) for cam_param in c2ws]
        relative_poses = []

        if self.args.src_cam_path is not None:
            print(f"Using source camera path: {self.args.src_cam_path}")
            with open(self.args.src_cam_path, 'r') as file:
                src_cam_data = json.load(file)
            
            src_c2w = self.parse_matrix(src_cam_data)
            src_c2w = src_c2w.transpose(1,0)
            src_c2w = src_c2w[:, [1, 2, 0, 3]]
            src_c2w[:3, 1] *= -1.
            src_c2w[:3, 3] /= 100
            src_cam_param = Camera(src_c2w)
            base_cam_params = src_cam_param
        else:
            base_cam_params = tgt_cam_params[0]
            
        for i in range(len(tgt_cam_params)):
            if self.use_interval_pose and i != 0:
                relative_pose = self.get_relative_pose([tgt_cam_params[i-1], tgt_cam_params[i]])
                relative_poses.append(torch.as_tensor(relative_pose)[:,:3,:][1])
            else:
                relative_poses.append(torch.as_tensor(tgt_cam_params[i].c2w_mat)[:3,:])
        pose_embedding = torch.stack(relative_poses, dim=0)  # 21x3x4
        pose_embedding = rearrange(pose_embedding, 'b c d -> b (c d)')
        data['camera'] = pose_embedding.to(torch.bfloat16)
        return data
    

    def __len__(self):
        return len(self.path)

def parse_args():
    parser = argparse.ArgumentParser(description="ReCamMaster Inference")
    
    #### jho added >>>>
    ## data
    parser.add_argument("--num_frames", type=int, default=81)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--use_reverse", action='store_true')
    parser.add_argument("--use_interval_pose", action='store_true')
    parser.add_argument("--src_cam_path", type=str, default=None)
    parser.add_argument("--tgt_cam_path", type=str, default=None)
    parser.add_argument("--repeat_first_frame", type=int, default=None)
    
    ## model 
    parser.add_argument("--no_projector", action="store_true")
    parser.add_argument("--repeat_time_rope", action="store_true")
    parser.add_argument("--use_src_adapter", action="store_true")
    parser.add_argument("--src_adapter_rank", type=int, default=32)
    #### jho added <<<<

    parser.add_argument("--dataset_path", type=str, default="./example_test_data")
    parser.add_argument("--ckpt_path", type=str, default="./models/ReCamMaster/checkpoints/step20000.ckpt")
    parser.add_argument("--output_dir", type=str, default="./outputs/dummy")
    parser.add_argument("--dataloader_num_workers", type=int, default=0)
    parser.add_argument("--cam_type", type=str, default=1)
    parser.add_argument("--cfg_scale", type=float, default=5.0)
    parser.add_argument("--num_inference_steps", type=int, default=20)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    # 1. Load Wan2.1 pre-trained models
    model_manager = ModelManager(torch_dtype=torch.bfloat16, device="cpu")
    model_manager.load_models([
        "models/Wan-AI/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors",
        "models/Wan-AI/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth",
        "models/Wan-AI/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth",
    ])
    pipe = WanVideoReCamMasterPipeline.from_model_manager(model_manager, device="cuda")

    # 2. Initialize additional modules introduced in ReCamMaster
    dim=pipe.dit.blocks[0].self_attn.q.weight.shape[0]
    for block in pipe.dit.blocks:
        block.cam_encoder = nn.Linear(12, dim)
        block.cam_encoder.weight.data.zero_()
        block.cam_encoder.bias.data.zero_()
        if args.no_projector:
            pass
        else:
            block.projector = nn.Linear(dim, dim)
            block.projector.weight = nn.Parameter(torch.eye(dim))
            block.projector.bias = nn.Parameter(torch.zeros(dim))

    pipe.dit.set_args(repeat_time_rope=args.repeat_time_rope)
    if args.use_src_adapter:
        for block in pipe.dit.blocks:
            block.self_attn.set_src_adapter(args.src_adapter_rank)

    # 3. Load ReCamMaster checkpoint
    state_dict = torch.load(args.ckpt_path, map_location="cpu")
    pipe.dit.load_state_dict(state_dict, strict=True)
    pipe.to("cuda")
    pipe.to(dtype=torch.bfloat16)

    output_dir = os.path.join(args.output_dir, f"cam_type{args.cam_type}")
    concat_output_dir = os.path.join(args.output_dir+"concat", f"cam_type{args.cam_type}")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        os.makedirs(concat_output_dir)

    # 4. Prepare test data (source video, target camera, target trajectory)
    dataset = TextVideoCameraDataset(
        args.dataset_path,
        os.path.join(args.dataset_path, "metadata.csv"),
        args,
        num_frames=args.num_frames,
        height=args.height,
        width=args.width,
        use_reverse=args.use_reverse,
        use_interval_pose=args.use_interval_pose,
        tgt_cam_path=args.tgt_cam_path,
        repeat_first_frame=args.repeat_first_frame,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=False,
        batch_size=1,
        num_workers=args.dataloader_num_workers
    )

    # 5. Inference
    for batch_idx, batch in enumerate(dataloader):
        save_p = os.path.join(output_dir, f"video{batch_idx}.mp4")
        concat_save_p = os.path.join(concat_output_dir, f"video{batch_idx}.mp4")
        if os.path.exists(save_p):
            print(f"{save_p} exists, skip.")
            continue
        target_text = batch["text"]
        source_video = batch["video"]
        target_camera = batch["camera"]
        
        video = pipe(
            prompt=target_text,
            negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
            source_video=source_video,
            target_camera=target_camera,
            cfg_scale=args.cfg_scale,
            num_inference_steps=args.num_inference_steps,
            seed=0, tiled=False,
            num_frames=source_video.shape[2], height=args.height, width=args.width,
        )

        pil_source_video = pipe.tensor2video(source_video[0])
        concat = concat_videos_hori([pil_source_video, video])
        save_video(video, save_p, fps=30, quality=5)
        save_video(concat, concat_save_p, fps=30, quality=5)
        