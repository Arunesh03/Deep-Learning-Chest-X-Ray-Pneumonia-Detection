# Chest X-Ray Pneumonia Detection

A deep learning pipeline that classifies chest X-ray images as **NORMAL** or **PNEUMONIA** using transfer learning on **EfficientNetV2-S**, with Grad-CAM-based visual explainability for clinical interpretability.

## Overview

This project fine-tunes a pre-trained EfficientNetV2-S convolutional neural network to detect pneumonia from pediatric chest X-ray images. It is built around the well-known [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) dataset and includes:

- An end-to-end PyTorch training and evaluation pipeline
- Transfer learning with a frozen EfficientNetV2-S backbone
- Feature map and depth-slice visualisations of internal CNN activations
- Grad-CAM heatmaps to visualise which regions of an X-ray drove the model's prediction
- A simple single-image inference function for live predictions

## Dataset

- **Source:** [Kaggle - chest-xray-pneumonia](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) (downloaded via `kagglehub`)
- **Classes:** `NORMAL`, `PNEUMONIA`
- **Structure:** `chest_xray/{train,test}/{NORMAL,PNEUMONIA}`
- **Splits used in this notebook:**
  - Train: 80% of the original training folder
  - Validation: remaining 20% of the original training folder (seeded split, `seed=42`)
  - Test: the dataset's original held-out test folder (624 images)

## Model Architecture

| Component | Detail |
|---|---|
| Backbone | `EfficientNetV2-S` (ImageNet pre-trained, via `torchvision`) |
| Transfer learning strategy | Backbone frozen (`requires_grad=False`); only the classifier head is trained |
| Classifier head | Final `Linear` layer replaced to output 2 classes |
| Input size | 224 × 224 RGB |
| Normalization | ImageNet mean/std (`[0.485, 0.456, 0.406]`, `[0.229, 0.224, 0.225]`) |
| Loss function | Cross-Entropy Loss |
| Optimizer | Adam (`lr=1e-3`), applied to classifier parameters only |
| LR Schedule | Cosine annealing |
| Epochs | 5 |
| Batch size | 32 |

### Data Augmentation (training set only)
- Resize to 224×224
- Random horizontal flip
- Random rotation (±15°)

## Results

Final performance on the held-out **test set** (624 images):

| Metric | Score |
|---|---|
| Accuracy | **80.77%** |
| Precision | 78.85% |
| Recall (Sensitivity) | 94.62% |
| F1-Score | 86.01% |

**Per-class breakdown:**

| Class | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| NORMAL | 0.87 | 0.58 | 0.69 | 234 |
| PNEUMONIA | 0.79 | 0.95 | 0.86 | 390 |

**Training loss progression:**

| Epoch | Avg. Loss |
|---|---|
| 1 | 0.2802 |
| 2 | 0.2361 |
| 3 | 0.2277 |
| 4 | 0.2253 |
| 5 | 0.2209 |

### Interpretation
The model is tuned toward high **recall on pneumonia** (94.62%), meaning it rarely misses a true pneumonia case — a desirable property for a screening tool, since false negatives are clinically more costly than false positives. However, this comes at the cost of lower recall on NORMAL cases (58%), indicating a meaningful number of healthy X-rays are flagged as pneumonia (false positives). This trade-off should be considered before any real-world or clinical use, and would benefit from threshold tuning, class-weighted loss, or further fine-tuning of backbone layers.

## Explainability (Grad-CAM)

The notebook includes a custom **Grad-CAM** implementation that:
- Hooks into the last convolutional block of EfficientNetV2-S
- Computes gradients with respect to the predicted class
- Overlays a heatmap on the original X-ray to highlight the regions most influential to the model's decision

This is included to support clinical interpretability — i.e., verifying that the model is attending to lung regions rather than spurious artefacts (image borders, text markers, etc.).

## Project Structure

```
.
├── Chest_X_Ray_Pneumonia_Detection.ipynb   # Main notebook (data, training, eval, Grad-CAM)
└── pneumonia_model_final.pth               # Saved model weights (generated after training)
```

## Notebook Workflow

1. **Data download & setup** — fetches the dataset via `kagglehub`, builds `ImageFolder` datasets and `DataLoaders' for train/val/test.
2. **Sample visualisation** — displays sample images from each class/split.
3. **Model setup** — loads pre-trained EfficientNetV2-S, freezes backbone, replaces classifier head.
4. **Feature map visualisation** — depth-slice and filter visualisations of early conv layer activations.
5. **Grad-CAM utilities** — defines the gradient-based class activation mapping function.
6. **Training loop** — trains the classifier head for 5 epochs, saves weights, evaluates on the test set, and plots a confusion matrix.
7. **Standalone evaluation** — reloads the model and re-computes classification report, confusion matrix, and summary metrics.
8. **Live inference demo** — runs a single-image prediction with class probabilities.

## Setup

### Requirements

```bash
pip install torch torchvision kagglehub opencv-python-headless grad-cam scikit-learn seaborn matplotlib numpy
```

### Kaggle API Credentials
The dataset is downloaded via `kagglehub`, which requires a Kaggle account and API token (`kaggle.json`) to be configured in your environment. See [Kaggle's API documentation](https://www.kaggle.com/docs/api) for setup instructions.

### Running

1. Open `Chest_X_Ray_Pneumonia_Detection.ipynb` in Jupyter, Google Colab, or your preferred notebook environment.
2. Run all cells in order — the dataset will download automatically in the first cell.
3. A GPU is strongly recommended for training (the notebook auto-detects CUDA availability and falls back to CPU otherwise).
4. After training, the model weights are saved to `pneumonia_model_final.pth` in the working directory.

### Running Inference on a New Image

```python
predict_live_sample(model, "path/to/your/xray.jpeg")
```

This prints the predicted class (`NORMAL` or `PNEUMONIA`) along with confidence scores for both classes.

## Limitations & Disclaimer

- This model is trained on a **single public dataset** and has not been validated across multiple institutions, scanners, or patient populations.
- The dataset is sourced from **pediatric patients**, so performance may not generalise to adult chest X-rays.
- Class imbalance exists in the training data (more PNEUMONIA than NORMAL images), which likely contributes to the lower recall on the NORMAL class.
- **This project is for educational and research purposes only.** It is not a validated diagnostic tool and must not be used for actual clinical decision-making without rigorous clinical validation, regulatory approval, and physician oversight.



