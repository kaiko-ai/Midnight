# Kaiko midnight
Midnight - Training State-of-the-Art Pathology Foundation Models with Orders of Magnitude Less Data

This repository contains the official implementation for the paper titled "Training state-of-the-art pathology foundation models with orders of magnitude less data." Our approach achieves competitive performance compared to leading pathology foundation models (FMs), despite being trained on significantly fewer whole slide images (WSIs).

## Overview

We propose a refined self-supervised training framework based on DINOv2 with modifications that optimize model performance specifically for computational pathology. Our main contributions include:

- Three novel pathology FMs trained with significantly reduced data (up to 100x fewer WSIs).
- Introduction of high-resolution post-training to enhance embedding quality.
- Ablation studies illustrating the impact of each proposed modification.

## Model Highlights

- **Midnight-12k**: Trained exclusively on the publicly available TCGA dataset (12k WSIs).
- **Midnight-92k**: Trained on TCGA and an additional proprietary dataset (PRV-80k).
- **Midnight-92k/392**: Our top-performing model fine-tuned with high-resolution post-training.

## Dataset

| Dataset | WSIs | Source        |
|---------|------|---------------|
| TCGA    | 12k  | Public        |
| PRV-80k | 80k  | Proprietary   |
| GTEx    | 25k  | Public        |
| CPTAC   | 7.2k | Public        |

## Key Techniques

- **Online patching**: Efficient real-time extraction of informative tiles.
- **Color augmentation (HED)**: Robustness to stain variations.
- **HSV filtering**: Removal of low-informative tissue regions.
- **KDE regularizer**: Ensures embedding diversity and stability.

## Evaluation

We comprehensively evaluated the models using two sets of open-source benchmarks:

- [eva](https://github.com/kaiko-ai/eva): tile and slide-level tasks (classification, segmentation).
- [HEST](https://github.com/mahmoodlab/HEST): gene expression prediction tasks.

Our best model **Midnight-92k/392** consistently outperforms or matches leading models like Virchow2 and UNI-2.

## Results Summary

| Model                                                          | AVG. | PCam 10&#160;shots | BACH | BRCS | BreaKHis | CRC  | Gleason | MHIST | PCam | Camelyon16 (small) | Panda (small) | CoNSeP | MoNuSAC | HEST |
|----------------------------------------------------------------|---------|-------------|------|------|----------|------|---------|-------|------|--------------------|---------------|--------|---------|------------|
| **[Midnight-92k/392](#usage)**       | **0.779** | **0.900** | **0.904** | **0.646** | 0.802     | 0.966     | **0.807** | 0.828     | **0.951** | 0.883     | 0.651     | **0.662** | **0.708** | 0.415     |
| [UNI-2](https://huggingface.co/MahmoodLab/UNI2-h)                  | **0.777** | **0.885** | **0.924** | **0.651** | **0.863** | **0.970** | 0.777     | 0.829     | **0.951** | 0.884     | **0.666** | 0.626     | 0.644     | **0.431** |
| [Virchow2](https://huggingface.co/paige-ai/Virchow2)               | **0.769** | 0.835     | 0.890     | 0.633     | 0.818     | 0.966     | **0.791** | **0.865** | 0.938     | **0.890** | 0.655     | 0.640     | 0.674     | 0.403     |
| **[Midnight-92k](#usage)**           | 0.768     | **0.882** | 0.889     | 0.615     | 0.793     | **0.967** | **0.823** | 0.831     | 0.948     | 0.882     | 0.643     | 0.629     | 0.656     | **0.425** |
| **[Midnight-12k](#usage)**           | 0.761     | 0.803     | **0.907** | 0.639     | 0.840     | **0.967** | 0.790     | 0.815     | 0.931     | 0.855     | 0.648     | 0.625     | 0.664     | 0.412     |
| [H-Optimus-0](https://huggingface.co/bioptimus/H-optimus-0)        | 0.759     | 0.831     | 0.752     | 0.620     | 0.813     | 0.962     | 0.769     | **0.850** | 0.943     | **0.896** | **0.672** | **0.644** | **0.687** | **0.425** |
| [Kaiko-B8](https://github.com/kaiko-ai/towards_large_pathology_fms)| 0.757     | 0.799     | 0.876     | 0.641     | **0.842** | 0.960     | 0.761     | 0.830     | 0.920     | 0.847     | 0.650     | **0.644** | 0.686     | 0.391     |
| [Prov_GigaPath](https://github.com/prov-gigapath/prov-gigapath)    | 0.757     | 0.853     | 0.794     | 0.626     | **0.846** | 0.959     | 0.727     | 0.831     | 0.944     | 0.887     | 0.657     | 0.628     | **0.688** | 0.405     |
| [Hibou-L](https://huggingface.co/histai/hibou-L)                   | 0.753     | 0.825     | 0.792     | **0.643** | 0.767     | 0.954     | 0.766     | **0.850** | **0.949** | 0.866     | **0.667** | **0.646** | 0.668     | 0.397     |
| [UNI](https://huggingface.co/MahmoodLab/UNI)                       | 0.753     | 0.833     | 0.797     | 0.613     | 0.808     | 0.954     | 0.759     | 0.841     | 0.937     | **0.899** | 0.662     | 0.627     | 0.662     | 0.391     |
| [Phikon](https://huggingface.co/owkin/phikon)                      | 0.727     | 0.826     | 0.744     | 0.579     | 0.715     | 0.946     | 0.743     | 0.824     | 0.919     | 0.861     | 0.648     | 0.624     | 0.644     | 0.377     |
| [Phikon-v2](https://huggingface.co/owkin/phikon-v2)                | 0.722     | 0.756     | 0.737     | 0.607     | 0.725     | 0.953     | 0.753     | 0.796     | 0.900     | 0.867     | 0.634     | 0.626     | 0.645     | 0.391     |
| [Lunit](https://github.com/lunit-io/benchmark-ssl-pathology)       | 0.720     | 0.763     | 0.785     | 0.627     | 0.759     | 0.943     | 0.758     | 0.785     | 0.905     | 0.836     | 0.604     | 0.600     | 0.630     | 0.362     |
| [vitg14 (nat. img.)](https://github.com/facebookresearch/dinov2)   | 0.675     | 0.721     | 0.724     | 0.578     | 0.783     | 0.943     | 0.740     | 0.855     | 0.881     | 0.505     | 0.509     | 0.565     | 0.614     | 0.351     |
| [vitg14 (initial)](https://github.com/facebookresearch/dinov2)     | 0.493     | 0.652     | 0.474     | 0.413     | 0.425     | 0.754     | 0.459     | 0.578     | 0.763     | 0.532     | 0.304     | 0.462     | 0.432     | 0.166     |


## Usage

```python
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from huggingface_hub import login

login()  # login or use an access token

model = timm.create_model("hf-hub:kaiko-ai/midnight", pretrained=True)
transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
model.eval()
```


### Model Weights

Pre-trained weights for M-12k are publicly available via Hugging-Face under the MIT license: [Download link](https://huggingface.co/kaiko-ai/midnight/tree/main)


 ## Citation
 ```bibtex
 @article{KDK2025,
   title={Training state-of-the-art pathology foundation models with orders of magnitude less data},
   author={Mikhail Karasikov, Joost van Doorn, Nicolas Känzig, Melis Erdal Cesur, Hugo Horlings, Robert Berke, Fei Tang, Sebastian Otálora},
   year={2025},
   journal={arXiv preprint}
}
```

<br />

<div align="center">
  <img src="https://github.com/kaiko-ai/midnight/blob/main/docs/images/kaiko-logo.png?raw=true" width="200">
</div>
