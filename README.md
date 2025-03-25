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

Our best model **M-92k/392** consistently outperforms or matches leading models like Virchow2 and UNI-2.

## Results Summary

| Model                                                          | AVERAGE | PCam-10Shot | BACH | BRCS | BreaKHis | CRC  | Gleason | MHIST | PCam | Camelyon16 (small) | Panda (small) | CoNSeP | MoNuSAC | HEST (avg) |
|----------------------------------------------------------------|---------|-------------|------|------|----------|------|---------|-------|------|--------------------|---------------|--------|---------|------------|
| **[Midnight-92k/392](https://github.com/kaiko-ai/Midnight)** | **.779** | .90         | .90  | .65  | .80      | .97  | .81     | .83   | .95  | .88                | .65           | .66    | .71     | .415       |
| [UNI-2](https://huggingface.co/MahmoodLab/UNI2-h)                | .777   | .89         | .92  | .65  | .86      | .97  | .78     | .83   | .95  | .88                | .67           | .63    | .64     | .431       |
| [Virchow2](https://huggingface.co/paige-ai/Virchow2)           | .769   | .84         | .89  | .63  | .82      | .97  | .79     | .87   | .94  | .89                | .66           | .64    | .67     | .403       |
| **[Midnight-92k](https://github.com/kaiko-ai/Midnight)**         | .768   | .88         | .89  | .62  | .79      | .97  | .82     | .83   | .95  | .88                | .64           | .63    | .66     | .425       |
| **[Midnight-12k](https://github.com/kaiko-ai/Midnight)**         | .761   | .80         | .91  | .64  | .84      | .97  | .79     | .82   | .93  | .86                | .65           | .63    | .66     | .412       |
| [H-Optimus-0](https://huggingface.co/bioptimus/H-optimus-0)       | .759   | .83         | .75  | .62  | .81      | .96  | .77     | .85   | .94  | .90                | .67           | .64    | .69     | .425       |
| [Kaiko-B8](https://github.com/kaiko-ai/towards_large_pathology_fms)             | .757   | .80         | .88  | .64  | .84      | .96  | .76     | .83   | .92  | .85                | .65           | .64    | .69     | .391       |
| [Prov_GigaPath](https://github.com/prov-gigapath/prov-gigapath)   | .757   | .85         | .79  | .63  | .85      | .96  | .73     | .83   | .94  | .89                | .66           | .63    | .69     | .405       |
| [Hibou-L](https://huggingface.co/histai/hibou-L)               | .753   | .83         | .79  | .64  | .77      | .95  | .77     | .85   | .95  | .87                | .67           | .65    | .67     | .397       |
| [UNI](https://huggingface.co/MahmoodLab/UNI)                   | .753   | .83         | .80  | .61  | .81      | .95  | .76     | .84   | .94  | .90                | .66           | .63    | .66     | .391       |
| [Phikon](https://huggingface.co/owkin/phikon)               | .727   | .83         | .74  | .58  | .72      | .95  | .74     | .82   | .92  | .86                | .65           | .62    | .64     | .377       |
| [Phikon-v2](https://huggingface.co/owkin/phikon-v2)           | .722   | .76         | .74  | .61  | .73      | .95  | .75     | .80   | .90  | .87                | .63           | .63    | .65     | .391       |
| [Lunit](https://github.com/lunit-io/benchmark-ssl-pathology)               | .720   | .76         | .79  | .63  | .76      | .94  | .76     | .79   | .91  | .84                | .60           | .60    | .63     | .362       |
| [vitg14 (nat. img.)](https://github.com/facebookresearch/dinov2) | .675   | .72         | .72  | .58  | .78      | .94  | .74     | .86   | .88  | .51                | .51           | .57    | .61     | .351       |
| [vitg14 (initial)](https://github.com/facebookresearch/dinov2)   | .493   | .65         | .47  | .41  | .43      | .75  | .46     | .58   | .76  | .53                | .30           | .46    | .43     | .166       |

## Usage

### Installation

```bash
git clone <repo_url>
cd <repo_dir>
pip install -r requirements.txt
```

### Model Weights

Pre-trained weights for M-12k are publicly available under the MIT license at:
	•	Download link coming soon


 ## Citation
 ```bibtex
 @misc{KDK2025,
  title={Training state-of-the-art pathology foundation models with orders of magnitude less data},
  author={Mikhail Karasikov, Joost van Doorn, Nicolas Känzig, Melis Erdal Cesur, Hugo Horlings, Robert Berke, Fei Tang, Sebastian Otálora},
  year={2025},
  booktitle={arXiV}
}
```

<br />

<div align="center">
  <img src="https://github.com/kaiko-ai/midnight/blob/main/docs/images/kaiko-logo.png?raw=true" width="200">
</div>
