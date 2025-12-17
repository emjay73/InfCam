# installation
pip install --upgrade pip  # Enable PEP 660 support.
pip install -e ".[train]"

# generate text caption for videos
CUDA_VISIBLE_DEVICES=0 python make_txt.py \
  --dataset_dir "./sample_videos" \
  --save_root_dir ./results/sample_caption


# make csv files for metadata
CUDA_VISIBLE_DEVICES=0 python make_csv.py \
  --txt_dir "./results/sample_caption"
  --save_csv_p "./results/sample_caption.csv"