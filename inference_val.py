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

class Camera(object):
    def __init__(self, c2w):
        c2w_mat = np.array(c2w).reshape(4, 4)
        self.c2w_mat = c2w_mat
        self.w2c_mat = np.linalg.inv(c2w_mat)

class TextVideoCameraValDataset(torch.utils.data.Dataset):
    def __init__(self, base_path, metadata_path, args, max_num_frames=81, frame_interval=1, num_frames=81, height=480, width=832, is_i2v=False, camera_extrinsics_path=None, reverse_video=False):
        metadata = pd.read_csv(metadata_path)
        
        self.path = [os.path.join(base_path, file_name) for file_name in metadata["file_name"]]
        
        self.text = metadata["text"].to_list()
        
        self.max_num_frames = max_num_frames
        self.frame_interval = frame_interval
        self.num_frames = num_frames
        self.height = height
        self.width = width
        self.is_i2v = is_i2v
        self.args = args
        self.cam_type = self.args.cam_type
        self.camera_extrinsics_path = camera_extrinsics_path
        self.reverse_video = reverse_video
            
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

        if self.reverse_video:
            frames = frames[::-1]

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
        start_frame_id = torch.randint(0, self.max_num_frames - (self.num_frames - 1) * self.frame_interval, (1,))[0]
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

        # emjay modified ---------------------------------
        offset = np.array([
            [1, 0, 0, 0],
            # [1, 0, 0, 2.5],
            [0, 1, 0, 0],            
            # [0, 1, 0, 2.5],            
            [0, 0, 1, 0],
            # [0, 0, 1, 2.5],
            [0, 0, 0, 1]
            # frame 10 : [0.999518 0 -0.0310409 0] [-0 1 0 0] [0.0310409 0 0.999518 0] [3390 1380 252.422 1]
            # frame 1 : "[1 0 0 0] [-0 1 0 0] [0 -0 1 0] [3390 1380 240 1] ",
        ]) # cam type 7 frame 10
        ret_poses = [offset @ target_cam_c2w, ] + [offset @ abs2rel @ abs_c2w for abs_c2w in abs_c2ws[1:]]
        # original --------------------------------------
        # ret_poses = [target_cam_c2w, ] + [abs2rel @ abs_c2w for abs_c2w in abs_c2ws[1:]]
        # -----------------------------------------------------        
        ret_poses = np.array(ret_poses, dtype=np.float32)
        return ret_poses

    # emjay added --------------------------------
    def get_relative_pose_custom(self, cam_params_src, cam_params_trg):
        # abs_w2c_src = cam_params_src.w2c_mat
        abs_c2w_src = cam_params_src.c2w_mat 
        abs_w2c_trg = cam_params_trg.w2c_mat
        # abs_c2w_trg = cam_params_trg.c2w_mat
        # target_cam_c2w = np.array([
        #     [1, 0, 0, 0],
        #     [0, 1, 0, 0],
        #     [0, 0, 1, 0],
        #     [0, 0, 0, 1]
        # ])
        # abs2rel = target_cam_c2w @ abs_w2cs[0]
        ret_poses = abs_w2c_trg @ abs_c2w_src
        ret_poses = np.array(ret_poses, dtype=np.float32)
        return ret_poses
    # -------------------------------------------------

    def __getitem__(self, data_id):        

        try:
            path_src = self.path[data_id]
            text = self.text[data_id]
            
            print("loading src video from", path_src)
            video_src = self.load_video(path_src)
            
            # 비디오 로딩 실패 시 에러 발생
            if video_src is None:
                raise ValueError(f"Failed to load video, it is None.")

            num_frames = video_src.shape[1]

            # path_trg = os.path.join(path_src.rsplit("_", 2)[0], "static_video.mp4")
            path_trg = os.path.join(os.path.dirname(path_src), "videos", f"cam{int(self.cam_type):02d}.mp4")
            print("loading trg video from", path_trg)
            video_trg = self.load_video(path_trg)
            
            # 비디오 로딩 실패 시 에러 발생
            if video_trg is None:
                raise ValueError(f"Failed to load video, it is None.")

            # emjay comment out --------------------------------
            # assert num_frames == 81
            # ----------------------------------------------------            
            data = {"text": text, 
                    "video_src": video_src, "video_trg": video_trg, 
                    "path_trg": path_trg, "path_src": path_src}

            # emjay added --------------------------------
            # if self.camera_extrinsics_path is None:
            self.camera_extrinsics_path = os.path.join(os.path.dirname(path_src), "cameras/camera_extrinsic_refcam10.json")
                # vid_mapping_path = os.path.join(os.path.dirname(path), "../cameras", f"video_mapping.json")
                # with open(vid_mapping_path, 'r') as file:
                #     vid_mapping = json.load(file)
                # vid_mapping_start_idx = vid_mapping[f"cam{int(self.cam_type):02d}"]["start_idx"]
                # self.fps = vid_mapping[f"cam{int(self.cam_type):02d}"]["fps"]
                # self.num_frames = vid_mapping[f"cam{int(self.cam_type):02d}"]["num_frames"]
                
            # -------------------------------------------------
            with open(self.camera_extrinsics_path, 'r') as file:
                cam_data = json.load(file)

            cam_idx = list(range(num_frames))[::4]
            
            # emjay modified --------------------------------
            # traj = []
            # for idx in cam_idx:
            #     decrease = bool((idx // 81) %2)            
            #     idx = 80 - (idx % 81) if decrease else idx % 81
            #     traj.append(self.parse_matrix(cam_data[f"frame{idx}"][f"cam{int(self.cam_type):02d}"]) )
            # original --------------------------------
            traj = [self.parse_matrix(cam_data[f"frame{idx}"][f"cam{int(self.cam_type):02d}"]) for idx in cam_idx]
            # -------------------------------------------------

            

            traj = np.stack(traj).transpose(0, 2, 1)
            c2ws = []
            for c2w in traj:
                c2w = c2w[:, [1, 2, 0, 3]]
                c2w[:3, 1] *= -1.
                c2w[:3, 3] /= 100
                c2ws.append(c2w)
            tgt_cam_params = [Camera(cam_param) for cam_param in c2ws]
            relative_poses = []
            for i in range(len(tgt_cam_params)):
                # emjay modified --------------------------------                    
                # if 'val' in path:
                #     relative_pose = self.get_relative_pose([tgt_cam_params[(80-vid_mapping_start_idx)//4], tgt_cam_params[i]])
                #     relative_poses.append(torch.as_tensor(relative_pose)[:,:3,:][1])                    
                    
                # else:
                relative_poses.append(torch.as_tensor(tgt_cam_params[i].c2w_mat)[:3,:])

                # -------------------------------------------------

                # original --------------------------------
                # relative_pose = self.get_relative_pose([tgt_cam_params[0], tgt_cam_params[i]])
                # relative_poses.append(torch.as_tensor(relative_pose)[:,:3,:][1])
                # -------------------------------------------------

            pose_embedding = torch.stack(relative_poses, dim=0)
            pose_embedding = rearrange(pose_embedding, 'b c d -> b (c d)')
            data['camera'] = pose_embedding.to(torch.bfloat16)
            
            # emjay added --------------------------------
        
            spec_src = path_src.split('val/')[1].split('/')[0]
            focal_mm = int(spec_src.split('_')[0][1:]) #24
            focal_px = focal_mm * max(self.height, self.width) / 23.76 
            intrinsic_src = torch.tensor([focal_px, focal_px, self.width//2, self.height//2]).to(torch.bfloat16)            

            if "aug" in spec_src:
                spec_trg = path_trg.split('val/')[1].split('/')[1]
                focal_mm = int(spec_trg.split('_')[-1][1:]) #24
                focal_px = focal_mm * max(self.height, self.width) / 23.76 
                intrinsic_trg = torch.tensor([focal_px, focal_px, self.width//2, self.height//2]).to(torch.bfloat16)            
            else:
                intrinsic_trg = intrinsic_src
        
            #     focal_mm = 24
            #     # focal_px = focal_mm * 1280 / 23.76 # image resize 가 동작하지 않는다는 가정이 들어간 수식.
            #     focal_px = focal_mm * max(self.height, self.width) / 23.76 
            #     cam_intrinsic = torch.tensor([focal_px, focal_px, self.width//2, self.height//2]).to(torch.bfloat16)
            data['intrinsic_trg'] = intrinsic_trg
            data['intrinsic_src'] = intrinsic_src
            # -------------------------------------------------
            return data

        except Exception as e:
            # 에러 발생 시 경고 메시지 출력 후 None 반환
            path = self.path[data_id]
            print(f"Warning: Skipping data at index {data_id} ({path}) due to error: {e}")
            return None

    def __len__(self):
        return len(self.path)

def parse_args():
    parser = argparse.ArgumentParser(description="ReCamMaster Inference")
    # parser.add_argument("--camera_extrinsics_path", type=str, default=None)
    parser.add_argument("--reverse_video", action="store_true", default=False)
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="./example_test_data",
        help="The path of the Dataset.",
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="./models/ReCamMaster/checkpoints/step20000.ckpt",
        help="Path to save the model.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results",
        help="Path to save the results.",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help="Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process.",
    )
    parser.add_argument(
        "--cam_type",
        type=str,
        default=1,
    )
    parser.add_argument(
        "--cfg_scale",
        type=float,
        default=5.0,
    )
    # emjay added --------------------------------
    parser.add_argument(
        "--idx_modified_layers",
        nargs="+",
        type=int,
        default=[0, 20, 2, 1, 28, 3, 27, 12, 29],
        help="Indices of the DiT blocks to modify."
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
        "--lora_rank",
        type=int,
        default=64,
        help="LoRA rank.",
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=64,
        help="LoRA alpha.",
    )
    parser.add_argument(
        "--metadata_file_name",
        type=str,
        default="metadata_val_aug.csv",
    )
    parser.add_argument(
        "--no_warp",
        action="store_true",
        default=False,
        help="Whether to use no warp.",
    )
    
    # ----------------------------------------------
    args = parser.parse_args()
    return args

# emjay added --------------------------------
def collate_filter_none(batch):
    """
    DataLoader의 collate_fn으로 사용될 함수.
    batch (list): __getitem__ 결과의 리스트. [data, None, data, ...] 형태일 수 있음.
    """
    # 리스트에서 None이 아닌 항목만 필터링
    batch = [item for item in batch if item is not None]
    # 필터링 후 배치가 비어있으면 None 반환
    if not batch:
        return None
    # 유효한 데이터가 있으면 기본 collate 함수로 배치 생성
    return torch.utils.data.default_collate(batch)
# -------------------------------------------------
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
        block.projector = nn.Linear(dim, dim)
        block.projector.weight = nn.Parameter(torch.eye(dim))
        block.projector.bias = nn.Parameter(torch.zeros(dim))
    pipe.dit.set_args(repeat_time_rope=False)

    # 3. Load ReCamMaster checkpoint
    state_dict = torch.load(args.ckpt_path, map_location="cpu")
    pipe.dit.load_state_dict(state_dict, strict=True)
    pipe.to("cuda")
    pipe.to(dtype=torch.bfloat16)

    # output_dir = os.path.join(args.output_dir, f"cam_type{args.cam_type}")
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 4. Prepare test data (source video, target camera, target trajectory)
    dataset = TextVideoCameraValDataset(
        args.dataset_path,
        os.path.join(args.dataset_path, args.metadata_file_name),
        args,
        # emjay added --------------------------------
        max_num_frames=args.num_frames,
        num_frames=args.num_frames,
        height=args.height,
        width=args.width,
        # camera_extrinsics_path=args.camera_extrinsics_path,
        reverse_video=args.reverse_video,
        # ---------------------------------------------
        
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=False,
        batch_size=1,
        num_workers=args.dataloader_num_workers,
        # emjay added --------------------------------
        collate_fn=collate_filter_none  # 이 부분을 추가합니다.
        # ---------------------------------------------
    )
    import torch
    def run_make_function(): 
        return torch.empty((50 * 1024**3 // 4,), dtype=torch.float32, device="cuda")
    run_make_function()

    # 5. Inference
    for batch_idx, batch in enumerate(dataloader):
        # emjay modified --------------------------------
        save_ps = []
        for path_trg in batch["path_trg"]:
            save_p = os.path.join(output_dir, os.path.dirname(path_trg).split("val/")[1]) 
            os.makedirs(save_p, exist_ok=True)
            save_ps.append(os.path.join(save_p, os.path.basename(path_trg)))
        save_p = save_ps[0]
        # original --------------------------------
        # save_p = os.path.join(output_dir, f"video{batch_idx}.mp4")
        # -------------------------------------------------
        if os.path.exists(save_p):
            print(f"Skipping {save_p} because it already exists.")
            continue
        # batch가 None일 수 있으므로 건너뛰는 로직 추가
        if batch is None:
            print(f"Skipping a batch because it was empty after filtering invalid data.")
            continue

        target_text = batch["text"]
        source_video = batch["video_src"] 
        target_camera = batch["camera"]

        video = pipe(
            prompt=target_text,
            negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
            source_video=source_video,
            target_camera=target_camera,
            cfg_scale=args.cfg_scale,
            num_inference_steps=50,
            seed=0, tiled=True,
            num_frames=args.num_frames, height=args.height, width=args.width,
        )
        save_video(video, save_p, fps=15, quality=5)
