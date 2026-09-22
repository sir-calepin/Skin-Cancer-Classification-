# Skin Cancer Classification with Deep Learning

A PyTorch deep-learning project for classifying dermatoscopic skin lesion images from the HAM10000 dataset into seven diagnostic categories.

The project compares a custom convolutional neural network (CNN) trained from scratch with ResNet50 transfer-learning approaches. The selected final model is an **unweighted ResNet50 transfer-learning model**, which achieved approximately **80% test accuracy** in the final project work.

> **Important:** This repository is an educational machine-learning project. It is not a medical device, does not provide clinical diagnosis, and must not be used as a substitute for qualified medical evaluation.

## Project goal

The goal is to classify dermatoscopic skin lesion images into these seven HAM10000 categories:

| Code | Diagnostic category |
|---|---|
| `akiec` | Actinic keratosis / intraepithelial carcinoma |
| `bcc` | Basal cell carcinoma |
| `bkl` | Benign keratosis-like lesions |
| `df` | Dermatofibroma |
| `mel` | Melanoma |
| `nv` | Melanocytic nevi |
| `vasc` | Vascular lesions |

The dataset is severely imbalanced. In the project split, `nv` represents roughly 67% of the observations, while classes such as `df` and `vasc` are rare. Therefore, the project compares both overall accuracy and class-level precision, recall, F1 score, and confusion matrices.

## Results

| Model | Framework | Class weighting | Test accuracy | Interpretation |
|---|---|---:|---:|---|
| Custom CNN | PyTorch | No | About 72–73% | Learned the majority `nv` class well but had weak minority-class detection |
| ResNet50 transfer learning | PyTorch | Yes | About 64–67% | Improved recall for rare classes, but decreased overall accuracy |
| ResNet50 transfer learning | PyTorch | No | About 78–80% | Best overall model; selected as the final model |
| Custom CNN experiment | TensorFlow/Keras | Not applicable | About 45% | Additional exploratory comparison |
| MobileNetV2 experiment | TensorFlow/Keras | Class-weighted experiment | About 68% | Better than the TensorFlow CNN but below final ResNet50 |
| EfficientNetB0 experiment | TensorFlow/Keras | Class-weighted experiment | About 11% | Experiment did not converge successfully in the recorded configuration |

The project selected **unweighted ResNet50** because it produced the strongest overall accuracy and the clearest overall confusion-matrix diagonal. The weighted ResNet50 remains important because it demonstrated the trade-off between aggregate accuracy and minority-class recall.

## Key findings

- Transfer learning with pretrained ResNet50 outperformed the custom CNN baseline on overall test accuracy.
- Class weighting improved recall for rare lesion classes, including dermatofibroma (`df`), but reduced total accuracy.
- The unweighted ResNet50 was the strongest final choice for the project’s overall-accuracy objective.
- Severe class imbalance materially influenced predictions, especially for the dominant `nv` class.
- Accuracy alone is insufficient for this problem; per-class recall and the confusion matrix are essential when assessing rare lesion types.

## Repository structure

```text
skin-lesion-classification/
├── data/          # Dataset instructions and local-only raw/processed data
├── notebooks/     # Optional exploratory notebooks, not required to run scripts
├── src/           # Reusable data, model, training, evaluation, and utility code
├── scripts/       # Command-line training and evaluation entry points
├── app/           # Optional Gradio prediction application
├── models/        # Local checkpoints and final model weights; excluded from Git
├── reports/       # Locally generated plots, reports, and metrics; excluded from Git
└── references/    # Dataset and project documentation
```

## Setup

### 1. Clone the repository

```bash
git clone [https://github.com/YOUR-USERNAME/skin-lesion-classification.git](https://github.com/YOUR-USERNAME/skin-lesion-classification.git)
cd skin-lesion-classification
```

### 2. Create and activate a virtual environment

macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

For standard CPU installation:

```bash
pip install -r requirements.txt
```

For an NVIDIA GPU, install a PyTorch build compatible with your CUDA version by following the official PyTorch installation selector, then install the remaining packages:

```bash
pip install -r requirements.txt
```

### 4. Download the dataset

Download **Skin Cancer MNIST: HAM10000** from Kaggle:

```text
[https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)
```

Place its contents in this structure:

```text
data/raw/
├── HAM10000_metadata.csv
├── HAM10000_images_part_1/
│   ├── ISIC_0024306.jpg
│   ├── ...
│   └── ISIC_0034321.jpg
└── HAM10000_images_part_2/
    ├── ISIC_0027419.jpg
    ├── ...
    └── ISIC_0034319.jpg
```

The full dataset is intentionally excluded from Git because it is large and subject to its own usage terms.

## Reproducing experiments

All scripts use the same stratified 80/20 train-test split with `random_state=42`.

### Train the baseline CNN

```bash
python scripts/train_baseline_cnn.py
```

### Train ResNet50 with class weights

```bash
python scripts/train_resnet50_weighted.py
```

### Train the final ResNet50 model

```bash
python scripts/train_resnet50_final.py
```

The final-model script saves the best checkpoint to:

```text
models/final/resnet50_final.pth
```

It also creates local outputs such as:

```text
reports/figures/resnet50_final_training_curves.png
reports/figures/resnet50_final_confusion_matrix.png
reports/metrics/resnet50_final_classification_report.txt
reports/metrics/resnet50_final_metrics.json
```

### Evaluate a saved model

```bash
python scripts/evaluate_model.py \
  --checkpoint models/final/resnet50_final.pth \
  --model resnet50 \
  --name resnet50_final
```

## Run the Gradio demo

Train the final model first, or copy your locally trained model file to:

```text
models/final/resnet50_final.pth
```

Then run:

```bash
python app/gradio_demo.py
```

Open the local URL shown in the terminal and upload a dermatoscopic image. The app returns the top three predicted classes and their confidence scores.

## Implementation details

### Data preparation

- Metadata is loaded from `HAM10000_metadata.csv`.
- Image file paths are resolved across the two official HAM10000 image folders.
- Labels are encoded in alphabetical order:
  `akiec`, `bcc`, `bkl`, `df`, `mel`, `nv`, `vasc`.
- A stratified 80/20 split maintains the class distribution across training and test sets.
- Images are resized to 224 × 224 pixels.
- Training includes random horizontal flips and random rotations.
- Images use ImageNet normalization for compatibility with pretrained ResNet50 weights.

### Model architectures

#### Custom CNN

The baseline model uses three convolutional blocks with 32, 64, and 128 channels, max pooling, dropout, and a two-layer classifier head.

#### ResNet50 transfer learning

The ResNet50 models begin with ImageNet pretrained weights. The pretrained backbone is frozen initially, and the original fully connected layer is replaced with:

```text
Linear(2048 → 256) → ReLU → Dropout(0.5) → Linear(256 → 7)
```

### Class weighting experiment

The weighted ResNet50 uses `CrossEntropyLoss(weight=class_weights)`, where the class weights are calculated from the training split. This gives more loss weight to rare classes and less weight to the majority `nv` class.

## Limitations

- The dataset is imbalanced, which can make overall accuracy look stronger than rare-class performance.
- The final training configuration freezes the pretrained ResNet50 backbone; additional fine-tuning may improve performance.
- Results can vary by hardware, package version, random initialization, augmentation, and training duration.
- Predictions are research outputs only and are not clinically validated.

## Future improvements

- Fine-tune selected upper ResNet50 layers with a small learning rate.
- Compare focal loss, weighted sampling, and data augmentation for minority classes.
- Add cross-validation or a validation split separate from the final test set.
- Add calibration analysis and confidence thresholding.
- Add explainability methods such as Grad-CAM.
- Evaluate on an external dataset before making any generalization claims.

## License

This repository uses the MIT License for the code. The HAM10000 dataset remains subject to its own dataset license and terms of use.

## Acknowledgments

- HAM10000 dataset: Skin Cancer MNIST: HAM10000 on Kaggle.
- PyTorch and TorchVision for model development and pretrained ResNet50 weights.
- scikit-learn for splitting, class weights, and evaluation metrics.
- Gradio for the optional local inference interface.
