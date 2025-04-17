# Kaiko midnight
Midnight - Training State-of-the-Art Pathology Foundation Models with Orders of Magnitude Less Data

This repository contains the official implementation for the paper titled "[Training state-of-the-art pathology foundation models with orders of magnitude less data](https://arxiv.org/abs/2504.05186v1)." Our approach achieves competitive performance compared to leading pathology foundation models (FMs), despite being trained on significantly fewer whole slide images (WSIs).

## Overview

We propose a refined self-supervised training framework based on DINOv2 with modifications that optimize model performance specifically for computational pathology. Our main contributions include:

- Three novel pathology FMs trained with significantly reduced data (up to 100x fewer WSIs).
- Introduction of high-resolution post-training to enhance embedding quality.

## Model Highlights

- **Midnight-12k**: Trained exclusively on the publicly available TCGA dataset (12k WSIs).
- **Midnight-92k**: Trained on TCGA and an additional proprietary dataset (NKI-80k).
- **Midnight-92k/392**: Our top-performing model fine-tuned with high-resolution post-training.

## Training Datasets

| Dataset | WSIs | Source        | Comment    | 
|---------|------|---------------|------------|
| TCGA    | 12k  | Public        | FFPE only  |
| NKI-80k | 80k  | Proprietary   | 10,141 patients, 31 organs |

## Training Components 

- **DINOv2**: Self-supervised training with [DINOv2](https://github.com/facebookresearch/dinov2).
- **[KDE regularizer](https://proceedings.mlr.press/v119/wang20k/wang20k.pdf)**: Replaced KoLeo in DINOv2 to ensure embedding diversity and training stability.
- **[Online patching](https://arxiv.org/pdf/2404.15217)**: Efficient real-time extraction of informative tiles.
- **Color augmentation ([HED](https://arxiv.org/pdf/1902.06543))**: Robustness to stain variations.
- **Tile [filtering](https://arxiv.org/html/2408.00738v3#S5)**: Removal of low-informative tissue regions.

## Evaluation

We comprehensively evaluated the models using two sets of open-source benchmarks:

- [eva](https://github.com/kaiko-ai/eva): For both tile (classification, segmentation) and slide-level tasks.
- [HEST](https://github.com/mahmoodlab/HEST): For gene expression prediction tasks (regression).

Our best model **Midnight-92k/392** consistently outperforms or matches leading models like Virchow2 and UNI-2.

## Results Summary

| Model | AVG. | PCam 10 shots | BACH | BRACS | BreaKHis | CRC | Gleason | MHIST | PCam | Cam16 (small) | Panda (small) | CoNSeP | MoNuSAC | HEST |
|-------|------|---------------|------|-------|----------|-----|---------|-------|------|---------------|---------------|--------|---------|------|
| **[Midnight-92k/392](#usage)** | **0.778** | **0.900** | **0.904** | **0.646** | 0.802 | 0.966 | **0.807** | 0.828 | **0.951** | 0.868 | 0.651 | **0.662** | **0.708** | 0.415 |
| [UNI-2](https://huggingface.co/MahmoodLab/UNI2-h) | **0.776** | **0.885** | **0.924** | **0.651** | **0.863** | **0.970** | 0.777 | 0.829 | **0.951** | **0.873** | **0.666** | 0.626 | 0.644 | **0.431** |
| **[Midnight-92k](#usage)** | **0.767** | **0.882** | 0.889 | 0.615 | 0.793 | **0.967** | **0.823** | 0.831 | 0.948 | **0.872** | 0.643 | 0.629 | 0.656 | **0.425** |
| [Virchow2](https://huggingface.co/paige-ai/Virchow2) | 0.766 | 0.835 | 0.890 | 0.633 | 0.818 | 0.966 | **0.791** | **0.865** | 0.938 | 0.860 | 0.646 | 0.640 | 0.674 | 0.403 |
| **[Midnight-12k](#usage)** | 0.763 | 0.803 | **0.907** | 0.639 | 0.840 | **0.967** | 0.790 | 0.815 | 0.931 | **0.869** | 0.656 | 0.625 | 0.664 | 0.412 |
| [Kaiko-B8](https://github.com/kaiko-ai/towards_large_pathology_fms) | 0.757 | 0.799 | 0.876 | 0.641 | **0.842** | 0.960 | 0.761 | 0.830 | 0.920 | 0.836 | 0.650 | **0.644** | 0.686 | 0.391 |
| [H-Optimus-0](https://huggingface.co/bioptimus/H-optimus-0) | 0.755 | 0.831 | 0.752 | 0.620 | 0.813 | 0.962 | 0.769 | **0.850** | 0.943 | 0.847 | **0.672** | **0.644** | **0.687** | **0.425** |
| [Prov_GigaPath](https://github.com/prov-gigapath/prov-gigapath) | 0.752 | 0.853 | 0.794 | 0.626 | **0.846** | 0.959 | 0.727 | 0.831 | 0.944 | 0.812 | 0.657 | 0.628 | **0.688** | 0.405 |
| [Hibou-L](https://huggingface.co/histai/hibou-L) | 0.751 | 0.825 | 0.792 | **0.643** | 0.767 | 0.954 | 0.766 | **0.850** | **0.949** | 0.852 | 0.654 | **0.646** | 0.668 | 0.397 |
| [UNI](https://huggingface.co/MahmoodLab/UNI) | 0.749 | 0.833 | 0.797 | 0.613 | 0.808 | 0.954 | 0.759 | 0.841 | 0.937 | 0.854 | **0.662** | 0.627 | 0.662 | 0.391 |
| [Phikon](https://huggingface.co/owkin/phikon) | 0.724 | 0.826 | 0.744 | 0.579 | 0.715 | 0.946 | 0.743 | 0.824 | 0.919 | 0.822 | 0.648 | 0.624 | 0.644 | 0.377 |
| [Phikon-v2](https://huggingface.co/owkin/phikon-v2) | 0.718 | 0.756 | 0.737 | 0.607 | 0.725 | 0.953 | 0.753 | 0.796 | 0.900 | 0.807 | 0.634 | 0.626 | 0.645 | 0.391 |
| [Lunit](https://github.com/lunit-io/benchmark-ssl-pathology) | 0.714 | 0.763 | 0.785 | 0.627 | 0.759 | 0.943 | 0.758 | 0.785 | 0.905 | 0.759 | 0.604 | 0.600 | 0.630 | 0.362 |
| [vitg14 (nat. img.)](https://github.com/facebookresearch/dinov2) | 0.674 | 0.721 | 0.724 | 0.578 | 0.783 | 0.943 | 0.740 | **0.855** | 0.881 | 0.500 | 0.509 | 0.565 | 0.614 | 0.351 |
| [vitg14 (initial)](https://github.com/facebookresearch/dinov2) | 0.493 | 0.652 | 0.474 | 0.413 | 0.425 | 0.754 | 0.459 | 0.578 | 0.763 | 0.526 | 0.304 | 0.462 | 0.432 | 0.166 |

## Model Weights
- **Midnight-12k**: Publicly available at https://huggingface.co/kaiko-ai/midnight.
- **Midnight-92k** & **Midnight-92k/392**: Trained on proprietary data and, hence, subject to restricted access.


## Usage

**Midnight-12k** is publicly available at [https://huggingface.co/kaiko-ai/midnight](https://huggingface.co/kaiko-ai/midnight).

Our models are trained on 224x224 images normalized with a mean of (0.5, 0.5, 0.5) and a standard deviation of (0.5, 0.5, 0.5). Please ensure you apply these exact normalization parameters when preparing your datasets for embedding extraction.

```python
from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import requests
from torchvision.transforms import v2

url = 'https://upload.wikimedia.org/wikipedia/commons/8/80/Breast_DCIS_histopathology_%281%29.jpg'
image = Image.open(requests.get(url, stream=True).raw)

transform = v2.Compose(
    [
        v2.Resize(224),
        v2.CenterCrop(224),
        v2.ToTensor(),
        v2.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ]
)
model = AutoModel.from_pretrained('kaiko-ai/midnight')
```

### Extract embeddings for classification
For segmentation tasks, the model output corresponds to 16x16 patch tokens (derived from 224/14=16).
```python
import torch

def extract_classification_embedding(tensor):
    cls_embedding, patch_embeddings = tensor[:, 0, :], tensor[:, 1:, :]
    return torch.cat([cls_embedding, patch_embeddings.mean(1)], dim=-1)

batch = transform(image).unsqueeze(dim=0)
embedding = extract_classification_embedding(model(batch).last_hidden_state)
print(f"Embedding shape: {embedding[0].shape}")
```

### Extract embeddings for segmentation

```python
import math
import torch

def extract_segmentation_embedding(tensor):
    features = tensor[:, 1:, :].permute(0, 2, 1)
    batch_size, hidden_size, patch_grid = features.shape
    height = width = int(math.sqrt(patch_grid))
    return features.view(batch_size, hidden_size, height, width)

batch = transform(image).unsqueeze(dim=0)
embedding = extract_segmentation_embedding(model(batch).last_hidden_state)
print(f"Embedding shape: {embedding[0].shape}")
```

### Use via Trident

Midnight-12k is now supported in the [Trident toolkit](https://github.com/mahmoodlab/TRIDENT), see the documentation for more details.

 ## Citation
 ```bibtex
 @misc{KDK2025,
   title={Training state-of-the-art pathology foundation models with orders of magnitude less data},
   author={Mikhail Karasikov and Joost van Doorn and Nicolas Känzig and Melis Erdal Cesur and Hugo Mark Horlings and Robert Berke and Fei Tang and Sebastian Otálora},
   year={2025},
   eprint={2504.05186},
   archivePrefix={arXiv},
   primaryClass={cs.CV},
   url={https://arxiv.org/abs/2504.05186}, 
}
```

<br />

<div align="center">
  <img src="https://github.com/user-attachments/assets/7848aee0-12a4-439b-97cb-d69b034b710c?raw=true" width="200">
</div>
