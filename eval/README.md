# Evaluation

### Running HEST

1. Install HEST and get the HEST-1k data according to the HEST's manual
2. Checkout the submodules `git submodule update --init --recursive`
3. Install hestcore `pip install hestcore`
4. Run the benchmark: `python run_hest.py --config hest_bench_config.yaml`

### Running EVA
1. Checkout the submodules `git submodule update --init --recursive`
2. Copy the custom model definitions [kaiko.py](./eva/src/eva/vision/models/networks/backbones/pathology/kaiko.py) and other helpers to eva: `cp kaiko.py object_tools.py backbones.py renormalized.py ./eva/src/eva/vision/models/networks/backbones/pathology/; mv kaiko.py kaiko.py_;`
2. Install EVA: `pip install -e eva`
2. Run benchmarks [run_eva_internal.sh](./run_eva_internal.sh), e.g.:
```bash
MODEL_NAME="vitg14_Kaiko_Midnight_concat";
TASK="camelyon16_small";
MODEL_NAME=$MODEL_NAME \
IN_FEATURES=$(python get_dim.py $MODEL_NAME 1) \
OUTPUT_ROOT=/mnt/vast01/shared/experimental/pathology_fm/mikhail/data/eva/RESULTS_patience/${TASK}/${MODEL_NAME} \
EMBEDDINGS_ROOT=/dev/shm/mikhail/eva/EMBEDDINGS/${TASK}/${MODEL_NAME} \
DATA_ROOT=/mnt/vast01/shared/experimental/pathology_fm/mikhail/data/eva/${TASK} \
NORMALIZE_MEAN=[0.5,0.5,0.5] \
NORMALIZE_STD=[0.5,0.5,0.5] \
python -m eva predict_fit --config configs/vision/pathology/offline/classification/${TASK}.yaml
```
