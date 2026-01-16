# <p align=center> PAGMFR: Collaborative Purification in Dual Spaces for Industrial Anomaly Detection via Hierarchical Reconstruction </p>
#### Kaiyue Wang &dagger;, Chengyan Qin &dagger;, Chenglizhao Chen, Jieru Chi, and Teng Yu*</sup>

<img width="11035" height="5782" alt="FIG01" src="https://github.com/user-attachments/assets/fe0a3e2e-ee14-4835-b351-e421ce334a9e" />

## Overview

PAGMFR (Pseudo-Anomaly Generation and Multi-scale Feature Reconstruction) is an industrial image anomaly detection method that generates pseudo-anomaly samples and combines multi-scale feature reconstruction. The method adaptively generates pseudo-anomalies for both texture and object categories, and detects real anomalies by comparing differences between normal and abnormal feature reconstruction.

## Quick Start

### Environment Configuration

**1. Create virtual environment (optional):**

```bash
conda create -n anomaly-detection python=3.7
conda activate anomaly-detection
```

**2. Install dependencies:**

```bash
# Basic dependencies
pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
pip install numpy==1.21.6 scipy==1.7.3 matplotlib==3.5.3 scikit-image==0.19.3
pip install pillow==9.1.1 tqdm==4.64.1 pandas==1.3.5 opencv-python==4.6.0.66
pip install scikit-learn==1.0.2

# PAGMFR specific dependencies
pip install noise  # For Perlin noise generation
```

### Data Preparation

1. **MVTec AD Dataset**
   - Visit [MVTec AD official website](https://www.mvtec.com/company/research/datasets/mvtec-ad)
   - Download and extract the dataset

2. **Texture Dataset (required for texture categories)**
   - You can use [DTD dataset](https://www.robots.ox.ac.uk/~vgg/data/dtd/) or custom texture images
   - Place texture images in a directory

3. **External Image Dataset (required for object categories)**
   - Image dataset containing various objects
   - Used for generating object-class pseudo-anomalies

4. **Directory Structure Example**
   ```
   /path/to/data/
   ├── mvtec_ad/          # MVTec AD dataset
   │   ├── bottle/
   │   ├── cable/
   │   └── ...
   ├── texture_dataset/   # Texture images
   │   ├── texture1.jpg
   │   ├── texture2.jpg
   │   └── ...
   └── external_images/   # External images
       ├── object1.jpg
       ├── object2.jpg
       └── ...
   ```

### Training Execution

1. **Configure parameters:**  
   Modify parameters at the end of `test.py`:
   ```python
   root_dir = "path/to/mvtec_ad"              # MVTec AD dataset root directory
   category = "bottle"                        # Category to train
   texture_dataset_path = "path/to/texture_dataset"  # Texture dataset path
   external_img_dir = "path/to/external_images"      # External images directory
   ```

2. **Start training:**
   ```bash
   python test.py
   ```

### Inference Testing

After training, the model is automatically saved as `pagmfr_{category}_best.pth`. Use the following code for inference:

```python
# Load model
model = PAGMFRModel().to(DEVICE)
model.load_state_dict(torch.load(f"pagmfr_{category}_best.pth"))
model.eval()

# Get test data
test_loader = get_mvtec_dataloader(root_dir, category, train=False)

# Perform validation
i_auroc, p_auroc, pro = validate_pagmfr(model, test_loader)
print(f"Test Results - I-AUROC: {i_auroc:.4f}, P-AUROC: {p_auroc:.4f}, PRO: {pro:.4f}")
```

## Model Architecture

PAGMFR model consists of four main modules:

1. **Pseudo-Anomaly Generator**
   - Texture-class anomaly generation: Uses Perlin noise and texture image fusion
   - Object-class anomaly generation: Extracts object regions from external images
   - Adaptive generation: Automatically selects generation strategy based on category

2. **Multi-scale Feature Reconstruction Module (MFRM)**
   - Multi-layer feature decoder
   - Reconstructs original feature representations
   - Maintains feature space consistency

3. **Multi-scale Feature Fusion Module (MFFM)**
   - Fuses original and reconstructed features
   - Channel concatenation fusion strategy
   - Adaptive feature selection

4. **Anomaly Calculation Module**
   - Feature-level anomaly calculation (cosine similarity)
   - Image-level anomaly calculation (CIELAB color + structural differences)
   - MLP fusion for final anomaly map generation

## Training Process

### Loss Functions

1. **Reconstruction Loss:** MSE + SSIM loss
2. **Anomaly Detection Loss:** Focal Loss, balancing positive and negative samples

### Training Strategy

- Batch size: 32
- Learning rate: 0.001
- Optimizer: Adam with weight_decay=1e-5
- Learning rate scheduler: Cosine annealing
- Training epochs: 600

## Evaluation Metrics

- **Image-level AUROC (I-AUROC):** Evaluates anomaly detection performance at image level
- **Pixel-level AUROC (P-AUROC):** Evaluates anomaly localization accuracy at pixel level
- **Per-Region Overlap (PRO):** Overlap rate between predicted and ground truth anomaly regions

## Visualization

Visualization results are automatically generated during training:

- Original image
- Ground truth anomaly mask
- Predicted anomaly heatmap

Visualization results are saved in the current directory with naming format:
```
pagmfr_{category}_{image_name}.png
```

## Customization and Extensions

### 1. Adding New Pseudo-Anomaly Types
```python
def generate_new_anomaly(self, img, parameters):
    # Implement new anomaly generation logic
    # ...
    return pseudo_anomaly, anomaly_mask
```

### 2. Modifying Backbone Network
```python
# Use different pre-trained models
self.backbone = models.resnet50(pretrained=True)  # Use ResNet50
```

### 3. Adjusting Anomaly Calculation Strategy
```python
# Modify anomaly map fusion strategy
def forward(self, features, rec_features, img, rec_img):
    # Add new anomaly calculation metrics
    texture_diff = calculate_texture_difference(img, rec_img)
    anomaly_map = feat_anomaly_map + img_anomaly_map + 0.1 * texture_diff
    return anomaly_map
```

## Common Issues

1. **Pseudo-Anomaly Generation Failure**
   - Check texture dataset path
   - Ensure external image file formats are correct
   - Adjust Perlin noise parameters

2. **Training Overfitting**
   - Increase data augmentation
   - Use Dropout or regularization
   - Reduce model complexity

3. **High Memory Usage**
   - Reduce batch size
   - Use gradient accumulation
   - Enable mixed precision training

4. **Class Imbalance**
   - Adjust pos_weight parameter of Focal Loss
   - Use weighted sampling
   - Add data balancing strategies

## Experimental Results

### MVTec AD Dataset Performance
Our method achieves competitive results on the MVTec AD benchmark:

<img width="1062" height="605" alt="FIG02" src="https://github.com/user-attachments/assets/d47abe06-4ff6-4670-9c47-135282c8b40c" />


*Note: Complete results for all 15 categories are available in our paper.*

# TABLE I  
IMAGE-LEVEL AUROC ANOMALY DETECTION RESULTS ON THE MVTEC AD DATASET. **RED, GREEN** AND **BLUE NUMBERS INDICATE THE BEST, SECOND-BEST AND THIRD-BEST RESULTS, RESPECTIVELY.**

| Category / Method | GN [14] | US [32] | DA [33] | PS [16] | PM [22] | CP [15] | RD [13] | PAGMFR |
|---|---|---|---|---|---|---|---|---|
| **Textures:**    |    |    |    |    |    |    |    |    |
| Carpet    | 69.9    | 86.8    | 91.6    | 92.9    | 99.8    | 93.9    | 98.9    | 99.0   |
| Grid    | 70.8    | 95.7    | 81.0    | 94.6    | 96.7    | 100    | 100    | 99.8   |
| Leather    | 84.2    | 86.2    | 88.2    | 90.9    | 100    | 100    | 100    | 100    |
| Tile    | 79.4    | 88.2    | 99.1    | 97.8    | 98.1    | 94.6    | 99.3    | 99.4   |
| Wood    | 83.4    | 98.2    | 97.7    | 96.5    | 99.2    | 99.1    | 99.2    | 99.3   |
| Average    | 77.54    | 91.02    | 91.4    | 94.54    | 98.76    | 97.52    | 99.48    | 99.5   |

| **Objects:**    |    |    |    |    |    |    |    |    |
| Bottle    | 89.2    | 97.6    | 99.0    | 98.6    | 99.9    | 98.2    | 100    | 100    |
| Cable    | 75.7    | 84.4    | 86.2    | 90.3    | 92.7    | 81.2    | 95.0    | 97.9   |
| Capsule    | 73.2    | 76.7    | 86.1    | 76.7    | 91.3    | 98.2    | 96.3    | 96.7   |
| Hazelnut    | 78.5    | 92.1    | 93.1    | 92.0    | 93.1    | 98.2    | 96.3    | 100    |
| Metal Nut    | 70.0    | 75.8    | 82.0    | 94.0    | 98.7    | 99.9    | 100    | 99.3   |
| Pill    | 74.3    | 90.0    | 87.9    | 86.1    | 93.3    | 94.9    | 96.6    | 97.0   |
| Screw    | 74.6    | 98.7    | 94.9    | 81.3    | 85.8    | 88.7    | 97.0    | 97.6   |
| Toothbrush    | 65.3    | 99.2    | 95.3    | 100    | 96.1    | 99.4    | 96.6    | 99.5   |
| Transistor    | 79.2    | 87.6    | 81.8    | 91.5    | 97.4    | 96.1    | 96.7    | 98.5   |
| Zipper    | 74.5    | 85.9    | 91.9    | 97.9    | 90.3    | 99.9    | 98.5    | 98.7   |
| Average    | 75.5    | 81.9    | 89.8    | 90.8    | 93.9    | 95.5    | 97.3    | 98.5   |
| Total Average    | 76.52    | 86.46    | 90.6    | 92.67    | 96.33    | 96.51    | 98.39    | 99.0   |

*Note: PIXEL-LEVEL AUROC AND PRO results on MVTEC AD, as well as comprehensive experimental results on MVTEC 3D-AD DATASET and VISA DATASET are available in our paper.*

## Related Resources

- [MVTec AD Dataset](https://www.mvtec.com/company/research/datasets/mvtec-ad)
- [DTD Texture Dataset](https://www.robots.ox.ac.uk/~vgg/data/dtd/)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [OpenCV Documentation](https://docs.opencv.org/)

## Academic Applications

PAGMFR is suitable for research in:
- Industrial quality inspection
- Defect detection
- Anomaly localization
- One-class classification
- Self-supervised learning

## Tips for Better Performance

- **For texture categories:** Use high-quality texture datasets with diverse patterns
- **For object categories:** Include external images with various object types and backgrounds
- **Hyperparameter tuning:** Adjust learning rate and batch size based on your GPU memory
- **Data augmentation:** Add more augmentation techniques for better generalization

## Performance Optimization

To improve training speed and reduce memory usage:
- Use gradient checkpointing
- Implement mixed precision training
- Optimize data loading with pinned memory
- Distribute training across multiple GPUs

## Debugging Tools

Built-in debugging features:
- Automatic visualization of training samples
- Loss curve plotting (can be added)
- Gradient flow analysis
- Memory usage monitoring
