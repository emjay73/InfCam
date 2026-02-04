GPU_ID=7
SEED=0
POSE_INTERVAL=5
REF="ref40"

for CAM in {1..10}; do
    CUDA_VISIBLE_DEVICES=${GPU_ID} python inference_infcam.py \
        --cam_type ${CAM} \
	    --ckpt_path "models/InfCam/step35000.ckpt" \
        --camera_extrinsics_path "./sample_data/cameras/camera_extrinsics_extended_${REF}.json" \
        --output_dir "./results/sample_data/extreme_jump${POSE_INTERVAL}/${REF}" \
	    --dataset_path "./sample_data" \
        --metadata_file_name "metadata.csv" \
        --num_frames 81 --width 832 --height 480 \
        --pose_interval ${POSE_INTERVAL} \
        --num_inference_steps 20 \
	    --zoom_factor 1.0 \
        --k_from_unidepth \
	    --seed ${SEED}
done
