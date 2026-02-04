# for POSE_INTERVAL in {1..10}; do
for POSE_INTERVAL in {1.5,2,2.5,3,3.5,4,4.5}; do
    CUDA_VISIBLE_DEVICES=7 python data_analysis.py \
        --k_from_unidepth \
        --camera_extrinsics_path "sample_data/cameras/camera_extrinsics_extended_ref40.json" \
        --pose_interval ${POSE_INTERVAL} \
        --num_frames 81 \
        --height 480 \
        --width 832
done