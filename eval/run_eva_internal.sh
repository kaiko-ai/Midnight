# Main benchmarks
for MODEL_NAME in {vitg14_Kaiko_Midnight_concat,}; do
for TASK in {pcam_10shots,camelyon16_small,panda_small,}; do
    ray job submit --no-wait --runtime-env=<(echo "$ENV") \
        --working-dir . --entrypoint-resources='{"all_nodes": 0.001}' --entrypoint-num-gpus 1 -- bash -c "\
    rm -rf /dev/shm/mikhail/eva/EMBEDDINGS/${TASK}/${MODEL_NAME}; \
    MODEL_NAME=$MODEL_NAME \
    IN_FEATURES=$(python get_dim.py $MODEL_NAME 1) \
    OUTPUT_ROOT=/experimental/pathology_fm/mikhail/data/eva/RESULTS_patience/${TASK}/${MODEL_NAME} \
    EMBEDDINGS_ROOT=/dev/shm/mikhail/eva/EMBEDDINGS/${TASK}/${MODEL_NAME} \
    DATA_ROOT=/experimental/pathology_fm/mikhail/data/eva/${TASK} \
    NORMALIZE_MEAN=[0.5,0.5,0.5] \
    NORMALIZE_STD=[0.5,0.5,0.5] \
    python -m eva predict_fit --config configs/vision/pathology/offline/classification/${TASK}.yaml; \
    rm -rf /dev/shm/mikhail/eva/EMBEDDINGS/${TASK}/${MODEL_NAME}";
done

for TASK in {bach,crc,mhist,patch_camelyon,breakhis,bracs,gleason_arvaniti}; do
    ray job submit --no-wait --runtime-env=<(echo "$ENV") \
        --working-dir . --entrypoint-resources='{"all_nodes": 0.001}' --entrypoint-num-gpus 1 -- bash -c "\
    rm -rf /dev/shm/mikhail/eva/EMBEDDINGS/${TASK}/${MODEL_NAME}; \
    MODEL_NAME=$MODEL_NAME \
    IN_FEATURES=$(python get_dim.py $MODEL_NAME 1) \
    PATIENCE=12500 \
    OUTPUT_ROOT=/experimental/pathology_fm/mikhail/data/eva/RESULTS_patience/${TASK}/${MODEL_NAME} \
    EMBEDDINGS_ROOT=/dev/shm/mikhail/eva/EMBEDDINGS/${TASK}/${MODEL_NAME} \
    DATA_ROOT=/experimental/pathology_fm/mikhail/data/eva/${TASK} \
    NORMALIZE_MEAN=[0.5,0.5,0.5] \
    NORMALIZE_STD=[0.5,0.5,0.5] \
    python -m eva predict_fit --config configs/vision/pathology/offline/classification/${TASK}.yaml; \
    rm -rf /dev/shm/mikhail/eva/EMBEDDINGS/${TASK}/${MODEL_NAME}";
done

for TASK in {consep,monusac}; do
    ray job submit --no-wait --runtime-env=<(echo "$ENV") \
        --working-dir . --entrypoint-resources='{"all_nodes": 0.001}' --entrypoint-num-gpus 1 -- bash -c "\
    MODEL_NAME=$MODEL_NAME \
    IN_FEATURES=$(python get_dim.py $MODEL_NAME) \
    PATIENCE=12500 \
    OUTPUT_ROOT=/experimental/pathology_fm/mikhail/data/eva/RESULTS_patience/${TASK}/${MODEL_NAME} \
    DATA_ROOT=/experimental/pathology_fm/mikhail/data/eva/${TASK} \
    NORMALIZE_MEAN=[0.5,0.5,0.5] \
    NORMALIZE_STD=[0.5,0.5,0.5] \
    python -m eva fit --config configs/vision/pathology/online/segmentation/${TASK}.yaml";
done
done


# High-resolution benchmarks
for MODEL_NAME in {DINOv2_vitg14_nki-tcga_post_100_aspect_epoch_059_bicubic_concat_resize392,}; do
for TASK in {pcam_10shots,camelyon16_small,panda_small,}; do
    ray job submit --no-wait --runtime-env=<(echo "$ENV") \
        --working-dir . --entrypoint-resources='{"all_nodes": 0.001}' --entrypoint-num-gpus 1 -- bash -c "\
    rm -rf /dev/shm/mikhail/eva/EMBEDDINGS/${TASK}/${MODEL_NAME}; \
    MODEL_NAME=$MODEL_NAME \
    IN_FEATURES=$(python get_dim.py $MODEL_NAME 1) \
    RESIZE_DIM=392 \
    OUTPUT_ROOT=/experimental/pathology_fm/mikhail/data/eva/RESULTS_patience/${TASK}/${MODEL_NAME} \
    EMBEDDINGS_ROOT=/dev/shm/mikhail/eva/EMBEDDINGS/${TASK}/${MODEL_NAME} \
    DATA_ROOT=/experimental/pathology_fm/mikhail/data/eva/${TASK} \
    NORMALIZE_MEAN=[0.5,0.5,0.5] \
    NORMALIZE_STD=[0.5,0.5,0.5] \
    python -m eva predict_fit --config configs/vision/pathology/offline/classification/${TASK}.yaml; \
    rm -rf /dev/shm/mikhail/eva/EMBEDDINGS/${TASK}/${MODEL_NAME}";
done

for TASK in {bach,crc,mhist,patch_camelyon,breakhis,bracs,gleason_arvaniti}; do
    ray job submit --no-wait --runtime-env=<(echo "$ENV") \
        --working-dir . --entrypoint-resources='{"all_nodes": 0.001}' --entrypoint-num-gpus 1 -- bash -c "\
    rm -rf /dev/shm/mikhail/eva/EMBEDDINGS/${TASK}/${MODEL_NAME}; \
    MODEL_NAME=$MODEL_NAME \
    IN_FEATURES=$(python get_dim.py $MODEL_NAME 1) \
    RESIZE_DIM=392 \
    PATIENCE=12500 \
    OUTPUT_ROOT=/experimental/pathology_fm/mikhail/data/eva/RESULTS_patience/${TASK}/${MODEL_NAME} \
    EMBEDDINGS_ROOT=/dev/shm/mikhail/eva/EMBEDDINGS/${TASK}/${MODEL_NAME} \
    DATA_ROOT=/experimental/pathology_fm/mikhail/data/eva/${TASK} \
    NORMALIZE_MEAN=[0.5,0.5,0.5] \
    NORMALIZE_STD=[0.5,0.5,0.5] \
    python -m eva predict_fit --config configs/vision/pathology/offline/classification/${TASK}.yaml; \
    rm -rf /dev/shm/mikhail/eva/EMBEDDINGS/${TASK}/${MODEL_NAME}";
done

for TASK in {consep,monusac}; do
    ray job submit --no-wait --runtime-env=<(echo "$ENV") \
        --working-dir . --entrypoint-resources='{"all_nodes": 0.001}' --entrypoint-num-gpus 1 -- bash -c "\
    MODEL_NAME=$MODEL_NAME \
    IN_FEATURES=$(python get_dim.py $MODEL_NAME) \
    RESIZE_DIM=392 \
    PATIENCE=12500 \
    OUTPUT_ROOT=/experimental/pathology_fm/mikhail/data/eva/RESULTS_patience/${TASK}/${MODEL_NAME} \
    DATA_ROOT=/experimental/pathology_fm/mikhail/data/eva/${TASK} \
    NORMALIZE_MEAN=[0.5,0.5,0.5] \
    NORMALIZE_STD=[0.5,0.5,0.5] \
    python -m eva fit --config configs/vision/pathology/online/segmentation/${TASK}.yaml";
done
done


# Benchmarks for ablation studies
for MODEL_NAME in {DINOv2_vitb14_four,}; do
for TASK in {bach,crc,mhist,patch_camelyon,camelyon16_small,panda_small,breakhis,bracs,gleason_arvaniti}; do
    RAY_ADDRESS="http://localhost:8265" kray job submit --no-wait --runtime-env=<(echo "$ENV") \
        --working-dir . --entrypoint-resources='{"four_nodes_1": 0.001}' --entrypoint-num-gpus 1 -- bash -c "\
    rm -rf /experimental/pathology_fm/mikhail/data/eva/EMBEDDINGS/${TASK}/${MODEL_NAME}; \
    MODEL_NAME=$MODEL_NAME \
    IN_FEATURES=$(python get_dim.py $MODEL_NAME 1) \
    OUTPUT_ROOT=/experimental/pathology_fm/mikhail/data/eva/RESULTS/${TASK}/${MODEL_NAME} \
    EMBEDDINGS_ROOT=/dev/shm/mikhail/eva/EMBEDDINGS/${TASK}/${MODEL_NAME} \
    DATA_ROOT=/experimental/pathology_fm/mikhail/data/eva/${TASK} \
    NORMALIZE_MEAN=[0.5,0.5,0.5] \
    NORMALIZE_STD=[0.5,0.5,0.5] \
    python -m eva predict_fit --config configs/vision/pathology/offline/classification/${TASK}.yaml; \
    rm -r /dev/shm/mikhail/eva/EMBEDDINGS/${TASK}/${MODEL_NAME}";
done

for TASK in {consep,monusac}; do
    RAY_ADDRESS="http://localhost:8265" kray job submit --no-wait --runtime-env=<(echo "$ENV") \
        --working-dir . --entrypoint-resources='{"four_nodes_1": 0.001}' --entrypoint-num-gpus 1 -- bash -c "\
    MODEL_NAME=$MODEL_NAME \
    IN_FEATURES=$(python get_dim.py $MODEL_NAME) \
    OUTPUT_ROOT=/experimental/pathology_fm/mikhail/data/eva/RESULTS/${TASK}/${MODEL_NAME} \
    DATA_ROOT=/experimental/pathology_fm/mikhail/data/eva/${TASK} \
    NORMALIZE_MEAN=[0.5,0.5,0.5] \
    NORMALIZE_STD=[0.5,0.5,0.5] \
    python -m eva fit --config configs/vision/pathology/online/segmentation/${TASK}.yaml";
done
done
