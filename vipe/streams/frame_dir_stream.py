# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pathlib import Path

import cv2
import torch

from vipe.streams.base import ProcessedVideoStream, StreamList, VideoFrame, VideoStream

# emjay added ---------
import omegaconf
# -------------------


class FrameDirStream(VideoStream):
    """
    A video stream from a directory of frame images.
    This does not support nested iterations.
    """

    def __init__(self, path: Path, seek_range: range | None = None, name: str | None = None) -> None:
        super().__init__()
        if seek_range is None:
            seek_range = range(-1)

        self.path = path
        self._name = name if name is not None else path.name

        # Find all image files in the directory
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
        self.frame_files = []
        for ext in image_extensions:
            self.frame_files.extend(sorted(path.glob(f'*{ext}')))
            self.frame_files.extend(sorted(path.glob(f'*{ext.upper()}')))
        
        self.frame_files = sorted(list(set(self.frame_files)))
        
        if not self.frame_files:
            raise ValueError(f"No image files found in directory: {path}")

        # emjay modified ------------
        # Read metadata from second frame # we concat first frame from source video, which may have different size from other frames.
        second_frame = cv2.imread(str(self.frame_files[1]))
        if second_frame is None:
            raise ValueError(f"Could not read second frame: {self.frame_files[1]}")
        
        self._height, self._width = second_frame.shape[:2]
        # original ------------------
        # # Read metadata from first frame
        # first_frame = cv2.imread(str(self.frame_files[0]))
        # if first_frame is None:
        #     raise ValueError(f"Could not read first frame: {self.frame_files[0]}")
        
        # self._height, self._width = first_frame.shape[:2]
        # ----------------------------
        
        # Assume 30 fps for frame directories (this is just for compatibility)
        self._fps = 30.0
        _n_frames = len(self.frame_files)

        self.start = seek_range.start
        self.end = seek_range.stop if seek_range.stop != -1 else _n_frames
        self.end = min(self.end, _n_frames)
        self.step = seek_range.step
        self._fps = self._fps / self.step

    def frame_size(self) -> tuple[int, int]:
        return (self._height, self._width)

    def fps(self) -> float:
        return self._fps

    def name(self) -> str:
        return self._name

    def __len__(self) -> int:
        return len(range(self.start, self.end, self.step))

    def __iter__(self):
        self.current_frame_idx = -1
        return self

    def __next__(self) -> VideoFrame:
        self.current_frame_idx += 1
        
        if self.current_frame_idx >= self.end:
            raise StopIteration

        if self.current_frame_idx < self.start:
            return self.__next__()

        if (self.current_frame_idx - self.start) % self.step != 0:
            return self.__next__()

        # Load the frame
        frame_path = self.frame_files[self.current_frame_idx]
        frame = cv2.imread(str(frame_path))
        
        if frame is None:
            raise ValueError(f"Could not read frame: {frame_path}")

        # emjay added ------------
        if frame.shape[:2] != (self._height, self._width):
            frame = cv2.resize(frame, (self._width, self._height))
        # ----------------------------
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb = torch.as_tensor(frame).float() / 255.0
        frame_rgb = frame_rgb.cuda()

        return VideoFrame(raw_frame_idx=self.current_frame_idx, rgb=frame_rgb)


class FrameDirStreamList(StreamList):
    def __init__(self, base_path, frame_start: int, frame_end: int, frame_skip: int, cached: bool = False
            # emjay added ---------
            , name_path_parts: int = 6
            # -------------------
        ) -> None:
        super().__init__()
        
        # Handle both single path (str) and multiple paths (list)
        if isinstance(base_path, str):
            base_paths = [base_path]
        elif isinstance(base_path, list) or isinstance(base_path, omegaconf.listconfig.ListConfig):
            base_paths = base_path
        else:
            raise ValueError(f"base_path must be str or list, got {type(base_path)}")
        
        self.frame_directories = []
        
        # Process each base_path
        for bp in base_paths:
            base_path_obj = Path(bp)
            
            if not base_path_obj.exists():
                raise ValueError(f"Path not found: {base_path_obj}")
            
            if not base_path_obj.is_dir():
                raise ValueError(f"Path is not a directory: {base_path_obj}")
            
            # Find all directories containing images (recursively)
            image_dirs = self._find_image_directories(base_path_obj)
            self.frame_directories.extend(image_dirs)
                
        if not self.frame_directories:
            raise ValueError(f"No frame directories found in: {base_paths}")
            
        self.frame_range = range(frame_start, frame_end, frame_skip)
        self.cached = cached
        # emjay added ---------
        self.name_path_parts = name_path_parts
        # -------------------
    def _find_image_directories(self, root_path: Path) -> list[Path]:
        """
        Recursively find all directories containing image files.
        
        Logic:
        1. If root_path contains images directly → return [root_path]
        2. If root_path has subdirectories with images → return those subdirectories
        3. Recursively search all subdirectories for image-containing folders
        """
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        
        def has_images(directory: Path) -> bool:
            """Check if directory contains any image files"""
            if not directory.is_dir():
                return False
            for file in directory.iterdir():
                if file.is_file() and file.suffix.lower() in image_extensions:
                    return True
            return False
        
        def find_recursive(directory: Path) -> list[Path]:
            """Recursively find all directories with images"""
            result = []
            
            # Check if current directory has images
            if has_images(directory):
                result.append(directory)
                # If this directory has images, don't search subdirectories
                # (assume this is the leaf level we want)
                return result
            
            # If no images in current directory, search subdirectories
            for item in directory.iterdir():
                if item.is_dir():
                    # Recursively search subdirectories
                    result.extend(find_recursive(item))
            
            return result
        
        image_dirs = find_recursive(root_path)
        
        # Sort for consistent ordering
        return sorted(image_dirs)

    def __len__(self) -> int:
        return len(self.frame_directories)

    def __getitem__(self, index: int) -> VideoStream:
        # emjay modified ---------
               
        directory_path = self.frame_directories[index]
        
        # Create name from last N folder names joined by slash
        path_parts = directory_path.parts
        last_n_parts = path_parts[-self.name_path_parts:] if len(path_parts) >= self.name_path_parts else path_parts
        custom_name = "/".join(last_n_parts)
        
        stream: VideoStream = FrameDirStream(
            directory_path, 
            seek_range=self.frame_range,
            name=custom_name
        )
        # original ---------
        # stream: VideoStream = FrameDirStream(self.frame_directories[index], seek_range=self.frame_range)
        # -------------------

        if self.cached:
            stream = ProcessedVideoStream(stream, []).cache(desc="Loading frames", online=False)
        return stream

    def stream_name(self, index: int) -> str:
        return self.frame_directories[index].name

