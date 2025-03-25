# Kaiko midnight
Midnight - Training State-of-the-Art Pathology Foundation Models with Orders of Magnitude Less Data

This repository contains the official implementation for the paper titled "Training state-of-the-art pathology foundation models with orders of magnitude less data." Our approach achieves competitive performance compared to leading pathology foundation models (FMs), despite being trained on significantly fewer whole slide images (WSIs).

## Overview

We propose a refined self-supervised training framework based on DINOv2 with modifications that optimize model performance specifically for computational pathology. Our main contributions include:

- Three novel pathology FMs trained with significantly reduced data (up to 100x fewer WSIs).
- Introduction of high-resolution post-training to enhance embedding quality.
- Ablation studies illustrating the impact of each proposed modification.

## Model Highlights

- **M-12k**: Trained exclusively on the publicly available TCGA dataset (12k WSIs).
- **M-92k**: Trained on TCGA and an additional proprietary dataset (PRV-80k).
- **M-92k/392**: Our top-performing model fine-tuned with high-resolution post-training.

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

| Model           | pc10 | bach | brcs | bkhs | crc  | glsn | mhst | pc   | c16  | pnd  | cnsp | mnsc | hest | avg.  |
|-----------------|------|------|------|------|------|------|------|------|------|------|------|------|------|-------|
| **M-92k/392**   | .90  | .90  | .65  | .80  | .97  | .81  | .83  | .95  | .88  | .65  | .66  | .71  | .415 | **.779** |
| UNI-2           | .89  | .92  | .65  | .86  | .97  | .78  | .83  | .95  | .88  | .67  | .63  | .64  | .431 | .777  |
| Virchow2        | .84  | .89  | .63  | .82  | .97  | .79  | .87  | .94  | .89  | .66  | .64  | .67  | .403 | .769  |
| **M-92k**       | .88  | .89  | .62  | .79  | .97  | .82  | .83  | .95  | .88  | .64  | .63  | .66  | .425 | .768  |
| **M-12k**       | .80  | .91  | .64  | .84  | .97  | .79  | .82  | .93  | .86  | .65  | .63  | .66  | .412 | .761  |
| H-Optimus-0     | .83  | .75  | .62  | .81  | .96  | .77  | .85  | .94  | .90  | .67  | .64  | .69  | .425 | .759  |
| Kaiko-B8        | .80  | .88  | .64  | .84  | .96  | .76  | .83  | .92  | .85  | .65  | .64  | .69  | .391 | .757  |
| Prov_GigaPath   | .85  | .79  | .63  | .85  | .96  | .73  | .83  | .94  | .89  | .66  | .63  | .69  | .405 | .757  |
| **M-12k (100M)**| .79  | .87  | .62  | .81  | .97  | .80  | .81  | .93  | .87  | .68  | .62  | .66  | .415 | .757  |
| Hibou-L         | .83  | .79  | .64  | .77  | .95  | .77  | .85  | .95  | .87  | .67  | .65  | .67  | .397 | .753  |
| UNI             | .83  | .80  | .61  | .81  | .95  | .76  | .84  | .94  | .90  | .66  | .63  | .66  | .391 | .753  |
| Phikon          | .83  | .74  | .58  | .72  | .95  | .74  | .82  | .92  | .86  | .65  | .62  | .64  | .377 | .727  |
| Phikon-v3       | .76  | .74  | .61  | .73  | .95  | .75  | .80  | .90  | .87  | .63  | .63  | .65  | .391 | .722  |
| Lunit           | .76  | .79  | .63  | .76  | .94  | .76  | .79  | .91  | .84  | .60  | .60  | .63  | .362 | .720  |
| vitg14 (nat. img.)| .72| .72  | .58  | .78  | .94  | .74  | .86  | .88  | .51  | .51  | .57  | .61  | .351 | .675  |
| vitg14 (initial)| .65  | .47  | .41  | .43  | .75  | .46  | .58  | .76  | .53  | .30  | .46  | .43  | .166 | .493  |


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

