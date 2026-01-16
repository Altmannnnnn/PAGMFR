# <p align=center> PAGMFR: Collaborative Purification in Dual Spaces for Industrial Anomaly Detection via Hierarchical Reconstruction </p>
#### Kaiyue Wang\textsuperscript{\dag}, Chengyan Qin\textsuperscript{\dag}, Chenglizhao Chen~\IEEEmembership{Member, IEEE}, Jieru Chi~\IEEEmembership{Member, IEEE}, and Teng Yu\,\textsuperscript{*}~\IEEEmembership{Member, IEEE}</sup>

 
 
<img width="12800" height="7200" alt="FIG1_01" src="https://github.com/user-attachments/assets/5cbbd032-7e70-40fe-9b0f-1744ee1538e2" />

 
 ## Highlight

This study proposes a novel knowledge distillation framework **RGPKD (Reconstruction-Guided and Prompt-based Knowledge Distillation)** for unsupervised industrial anomaly detection. By integrating a **reconstruction-guided loss** and an **asymmetric prompt module** into the teacher-student architecture, our method effectively addresses the limitations of symmetric distillation, such as insufficient feature discrepancy in anomalous regions and detail loss during decoding. The key innovations include:

1. **Asymmetric distillation with reconstruction guidance** – enhances the student’s representation of normal patterns, thus amplifying feature differences under anomalies.
2. **Dynamic prompt module** – retrieves and fuses multi-scale contextual details from the teacher’s feature bank, improving boundary accuracy in anomaly localization.
3. **Balanced multi-task learning** – combines feature matching, reconstruction, and prompt fusion into a unified optimization framework.

## Introduction

Unsupervised anomaly detection based on knowledge distillation has shown great promise for industrial inspection, where labeled defect samples are scarce. However, existing methods typically adopt **symmetric teacher-student architectures**, which lead to **insufficient feature discrepancies** in subtle anomalous regions and **loss of fine details** during up-sampling in the decoder. To overcome these limitations, we propose **RGPKD**, a novel framework that introduces:

- **Reconstruction guidance**: The student network (U-Net style) is trained not only to match the teacher’s multi-scale features but also to reconstruct the input image, enforcing a stronger constraint for normal pattern learning.
- **Prompt-based detail injection**: A novel prompt module retrieves the most relevant multi-scale features from a prompt bank built from the teacher’s features and dynamically fuses them into the student’s decoding path, preserving fine-grained details.
- **Asymmetric architecture**: The teacher (ResNet18) and student (U-Net) are structurally heterogeneous, which enlarges the feature discrepancy between normal and anomalous regions, thereby improving detection sensitivity.

Experiments on MVTec AD and MVTec 3D-AD datasets demonstrate that RGPKD achieves state-of-the-art performance in both anomaly detection and localization, with superior generalization across texture and object categories.

## Usage

## Prerequisites

- [Python 3.5](https://www.python.org/)
- [Pytorch 1.3](http://pytorch.org/)
- [Numpy 1.15](https://numpy.org/)
- [Apex](https://github.com/NVIDIA/apex)
- [adversarial-robustness-toolbox](https://github.com/Trusted-AI/adversarial-robustness-toolbox)

## Clone repository

```shell
git clone https://github.com/Cariaaaaaa/RGPKD.git
```

## Dataset

Download the following datasets and unzip them into `data` folder

- ([MVTec Anomaly Detection Dataset: MVTec Software](https://www.mvtec.com/company/research/datasets/mvtec-ad))

## Dataset configuration

- For the training setup, update the `--dir_dataset` parameter in the `train.py` file to your training data path, e.g., `dir_dataset='./your_path/DUTS'`.
- For the testing, place all testing datasets in the same folder, and update the `--test_dataset_root` parameter in the `test.py` file to point to your testing data path, e.g., `test_dataset_root='./your_testing_data_path/'`.

## Training

```shell
    cd src/
    python3 train.py
```

- Warm-up and linear decay strategies are used to change the learning rate `lr`
- After training, the result models will be saved in `out` folder

## Testing

```shell
    cd src
    python3 test.py
```

- After testing, saliency maps of `PASCAL-S`, `ECSSD`, `HKU-IS`, `DUT-OMRON`, `DUTS-TE` will be saved in `./experiments/` folder.

## Result

Our method achieves outstanding performance on the MVTec AD benchmark. As shown in **Table I** (Image-level AUROC results), RGPKD outperforms existing state-of-the-art methods across nearly all categories, with an **average image-level AUROC of 99.8%**. Notably, it achieves **99.9% on texture categories** and **99.6% on object categories**, demonstrating robust generalization across different anomaly types.

<img width="1312" height="617" alt="image" src="https://github.com/user-attachments/assets/b6657c3f-e31b-4861-91ee-f9baa215b6ae" />



The method also excels in pixel-level localization, achieving high AUROC and PRO scores, which confirms its capability in precisely segmenting anomaly boundaries even in complex industrial scenes.

<img width="1283" height="660" alt="image" src="https://github.com/user-attachments/assets/9c0e0aab-40d7-4bab-b6db-008bac52824e" />

