import cv2
import os
import glob
import re
import shutil
import multiprocessing as mp
from functools import partial
from tqdm import tqdm

import argparse
parser = argparse.ArgumentParser()
# parser.add_argument("--input_dir", type=str, required=True)
# parser.add_argument("--src_dir", type=str, required=True)
parser.add_argument("--src_dir", type=str, default=None)
parser.add_argument("--parse_dir", type=str, required=True)
parser.add_argument("--save_first_frame", action='store_true')
parser.add_argument("--num_workers", type=int, default=8, help="Number of CPU workers for multiprocessing")
args = parser.parse_args()

# 출력 디렉토리
paths_gen = glob.glob(os.path.join(args.parse_dir, "*", "*.mp4"))

if args.src_dir is not None:
    paths_src = glob.glob(os.path.join(args.src_dir, "*", "*.mp4"))
else:
    if args.save_first_frame:
        raise ValueError("src_dir is required when save_first_frame is True")
    paths_src = None

output_dir_ = args.parse_dir

def process_single_video(video_path, paths_src, output_dir_):
    """단일 비디오를 프레임으로 변환하는 worker 함수"""
    try:
        video_folder = video_path.split("/")[-2]
        
        # emjay modified ------------------------------------------------
        raw_name = video_path.split("/")[-1]  # 예: "video1"
        match = re.match(r"(video)(\d+).mp4", raw_name)
        # original ------------------------------------------------
        # raw_name = video_path.split("/")[-1].split(".")[0]  # 예: "video1"
        # match = re.match(r"(video)(\d+)", raw_name)
        # ------------------------------------------------
        if match:
            prefix, number = match.groups()
            new_number = int(number)
            video_name = f"{prefix}{new_number:01d}"  # 0으로 패딩하려면 :02d 등 조절
            output_dir = os.path.join(output_dir_, video_folder, "frames", video_name)
        else:
            raise ValueError(f"Invalid video name format: {raw_name}")
        
        output_dir = os.path.join(output_dir_, video_folder, "frames", video_name)
        
        # 이미 존재하는 경우 스킵
        if os.path.exists(output_dir) and len(os.listdir(output_dir)) > 0:
            return f"Skipped {video_name} parsing frames (already exists)"
        else:
            os.makedirs(output_dir, exist_ok=True)
            print(f"Created {output_dir}")
        
        # emjay added ------------------------------------------------
        # load first frame from src video
        first_frame = None
        if args.save_first_frame:
            for src_video_path in paths_src:
                if raw_name in src_video_path:
                    cap = cv2.VideoCapture(src_video_path)
                    ret, frame = cap.read()
                    cap.release()
                    if ret:
                        first_frame = frame
                    break
        # ------------------------------------------------
        # 비디오 캡처 객체 생성
        cap = cv2.VideoCapture(video_path)
    
        frame_idx = 0
        if first_frame is not None:
            frame_filename = os.path.join(output_dir, f"{frame_idx:04d}.jpg")
            cv2.imwrite(frame_filename, first_frame)
            frame_idx = 1
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            height, width = frame.shape[:2]
            cropped_frame = frame[:, :] 
            
            # 프레임 저장 (0001.jpg, 0002.jpg, ...)
            frame_filename = os.path.join(output_dir, f"{frame_idx:04d}.jpg")
            cv2.imwrite(frame_filename, cropped_frame)
            
            frame_idx += 1
        
        cap.release()
        return f"Processed {video_name}: {frame_idx} frames"
        
    except Exception as e:
        return f"Error processing {video_path}: {str(e)}"

def main():
    print(f"Found {len(paths_gen)} videos to process")
    print(f"Using {args.num_workers} CPU workers")
    
    # worker 함수에 고정 인수들을 미리 바인딩
    worker_func = partial(process_single_video, 
                         paths_src=paths_src, 
                         output_dir_=output_dir_)
    
    try:
        # spawn 컨텍스트 사용 (안정성 향상)
        ctx = mp.get_context('spawn')
        
        with ctx.Pool(processes=args.num_workers) as pool:
            # tqdm으로 진행상황 표시
            results = list(tqdm(
                pool.imap(worker_func, paths_gen),
                total=len(paths_gen),
                desc="Processing videos"
            ))
        
        # 결과 출력
        print("\n=== Processing Results ===")
        for result in results:
            print(result)
            
    except Exception as e:
        print(f"Multiprocessing failed: {e}")
        print("Falling back to sequential processing...")
        
        # 순차 처리 fallback
        for video_path in tqdm(paths_gen, desc="Processing videos (sequential)"):
            result = process_single_video(video_path, paths_src, output_dir_)
            print(result)

if __name__ == "__main__":
    main()