# Evaluation

## Environment

```
# Mostly from the environmental setup of vipe ( https://github.com/nv-tlabs/vipe ).
# Create a new conda environment and install 3rd-party dependencies
git clone https://github.com/emjay73/InfCam.git traj_err
cd traj_err
git checkout traj_err
conda env create -f envs/base.yml
conda activate vipe
# You can switch to your own PyPI index if you want.
pip install -r envs/requirements.txt --extra-index-url https://download.pytorch.org/whl/cu128

# Build the project and install it into the current environment
# Omit the -e flag to install the project as a regular package
pip install --no-build-isolation -e .
```
And then install the following additional packages.
```
pip install pandas
```

## WebVid Samples Evaluation
Step1. prepare your source videos in the following format
```
DATA
└── webvid
    └── videos
        ├── video0.mp4
        ├── video1.mp4
        ...
        └── video99.mp4
```
Step2. prepare your resulting video and camera pose used to generate the video in the following format
```
results
└── exp_step20k_ref40
    ├── cam_type1
    │   ├── video0.mp4
    │   ├── 
    │   ...
    │   └── video99.mp4
    ├── cam_type2
    ...
    └── cam_type10
```
```
cameras
├── camera_extrinsics_ref0.json
└── camera_extrinsics_ref40.json
```

Step3. run vipe
```
# modify GPU_DEVICES, CAM_TYPES, PATH_SRC_DIR, PATH_GEN_DIR variables in the run_vipe_{data}.sh before run
# ex) GPU_DEVICES="7" # support only single gpu.
# ex) CAM_TYPES=(1 2 3 4 5 6 7 8 9 10)
# ex) DATASET_TYPE='webvid'
# ex) PATH_SRC_DIR="./DATA/webvid"
# ex) PATH_GEN_DIR="results/exp_step20k_ref40"

bash run_vipe_webvid.sh
```
'frames' folder will be created under each 'cam_typeX' directory. \
For those who experiencing 'ImportError: libGL.so.1: cannot open shared object file: No such file or directory'
```
pip uninstall opencv-python
pip install opencv-python-headless
```

Step4. run evaluation
```
# modify GPU_DEVICES, DATASET_TYPE, GT_JSON, SEQ_NAME variables in the run_metric.sh before run
# ex) GPU_DEVICES="0" 
# ex) DATASET_TYPE="webvid"
# ex) GT_JSON="cameras/camera_extrinsics_ref40.json"
# ex) SEQ_NAME="results/exp_step20k_ref40" # check vipe_results/vipe and write the subfolder path of interest.
bash run_metric_metric.sh
```

