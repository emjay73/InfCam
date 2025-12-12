
import hydra
from omegaconf import DictConfig

# emjay added ---------
import torch
import os


def preallocate_gpu_memory(gpu_id: int = None, memory_gb: float = None) -> torch.Tensor:
    """
    GPU 메모리를 미리 지정한 만큼 할당하는 함수
    
    Args:
        gpu_id: GPU ID (None이면 CUDA_VISIBLE_DEVICES 또는 기본 GPU 사용)
        memory_gb: 할당할 메모리 크기 (GB, None이면 환경변수에서 읽거나 기본값 사용)
    
    Returns:
        할당된 텐서 (메모리 해제를 위해서는 반환값을 유지해야 함)
    """
    if not torch.cuda.is_available():
        print("[WARNING] CUDA is not available. Skipping GPU memory preallocation.")
        return None
    
    # GPU ID 결정
    if gpu_id is None:
        if "CUDA_VISIBLE_DEVICES" in os.environ:
            # CUDA_VISIBLE_DEVICES가 설정되어 있으면 0번으로 인식됨
            gpu_id = 0
        else:
            gpu_id = 0
    
    # 메모리 크기 결정
    if memory_gb is None:
        memory_gb = float(os.environ.get("GPU_MEMORY_GB", "42.0"))
    
    device = torch.device(f"cuda:{gpu_id}")
    target_bytes = int(memory_gb * 1024**3)
    num_elements = target_bytes // 4  # float32 = 4 bytes
    
    print(f"[INFO] Pre-allocating {memory_gb:.2f}GB on {device}...")
    memory_tensor = torch.empty(num_elements, dtype=torch.float32, device=device)
    print(f"[INFO] GPU memory pre-allocated successfully. ({memory_gb:.2f}GB)")
    
    return memory_tensor
# -------------------------------

@hydra.main(version_base=None, config_path="configs", config_name="default_emjay")
def run(args: DictConfig) -> None:
    from vipe.streams.base import StreamList

    # Gather all video streams
    stream_list = StreamList.make(args.streams)

    from vipe.pipeline import make_pipeline
    from vipe.utils.logging import configure_logging

    # Process each video stream
    logger = configure_logging()

    # emjay added ---------
    pipeline = make_pipeline(args.pipeline)
    # ----------------------
    for stream_idx in range(len(stream_list)):
        video_stream = stream_list[stream_idx]
        logger.info(
            f"Processing {video_stream.name()} ({stream_idx + 1} / {len(stream_list)})"
        )
        # emjay commented out ---------
        # pipeline = make_pipeline(args.pipeline)
        # -------------------------------
        pipeline.run(video_stream)
        logger.info(f"Finished processing {video_stream.name()}")

if __name__ == "__main__":
    # emjay added ---------
    # GPU 메모리 사전 할당 (설정에서 읽거나 환경변수 사용)
    #gpu_id = getattr(args, "gpu_id", None)
    #memory_gb = getattr(args, "preallocate_gpu_memory_gb", None)
    #memory_tensor = preallocate_gpu_memory(gpu_id, memory_gb)
    # -------------------------------
    
    run()
    # main()
    # python glomap_final.py --gt_json /home/nas5/hoiyeongjin/DATA/MultiCamVideo-Dataset/val/10basic_trajectories/cameras/camera_extrinsics.json --image_dir /home/nas5/hoiyeongjin/repos/Research/nvvs/FRAMES/crop/cam10 --cam_type cam10
