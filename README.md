## Installation

```bash
# Enable PEP 660 support.
pip install --upgrade pip
pip install -e ".[train]"
```

## Usage

### 1. Generate Text Caption for Videos

```bash
CUDA_VISIBLE_DEVICES=0 python make_txt.py \
  --dataset_dir "./sample_videos" \
  --save_root_dir "./results/sample_caption"
```

### 2. Make CSV Files for Metadata

```bash
CUDA_VISIBLE_DEVICES=0 python make_csv.py \
  --txt_dir "./results/sample_caption" \
  --save_csv_p "./results/sample_caption.csv"
```