# Evaluation

### Running HEST

1. Get the submodules `git submodule update --init --recursive`
2. Install HEST `pip install -e HEST` and get the HEST-1k data according to the HEST's [manual](https://github.com/mahmoodlab/HEST/tree/568df6ce8e6cd88829866a8cfdb5d2452f72f96e?tab=readme-ov-file#downloadquery-hest-1k-1tb)
3. Install hestcore `pip install hestcore`
4. Run the benchmark: `python run_hest.py --config hest_bench_config.yaml`

### Running EVA
1. Get the submodules `git submodule update --init --recursive`
2. Copy the custom model definitions [kaiko.py](./eva/src/eva/vision/models/networks/backbones/pathology/kaiko.py) and other helpers to eva: `cp kaiko.py object_tools.py backbones.py renormalized.py ./eva/src/eva/vision/models/networks/backbones/pathology/; mv kaiko.py kaiko.py_;`
3. Install EVA: `pip install -e eva`
4. Download the data according following the eva [instructions](https://kaiko-ai.github.io/eva/main/datasets) (also see the [eva user guide](https://kaiko-ai.github.io/eva/main/user-guide/))
5. Run benchmarks [run_eva_internal.sh](./run_eva_internal.sh), e.g.:
```bash
MODEL_NAME="vitg14_Kaiko_Midnight_concat";
TASK="camelyon16_small";
MODEL_NAME=$MODEL_NAME \
IN_FEATURES=$(python get_dim.py $MODEL_NAME 1) \
OUTPUT_ROOT=/pathology_fm/mikhail/data/eva/RESULTS_patience/${TASK}/${MODEL_NAME} \
EMBEDDINGS_ROOT=/dev/shm/mikhail/eva/EMBEDDINGS/${TASK}/${MODEL_NAME} \
DATA_ROOT=/pathology_fm/mikhail/data/eva/${TASK} \
NORMALIZE_MEAN=[0.5,0.5,0.5] \
NORMALIZE_STD=[0.5,0.5,0.5] \
python -m eva predict_fit --config configs/vision/pathology/offline/classification/${TASK}.yaml
```
