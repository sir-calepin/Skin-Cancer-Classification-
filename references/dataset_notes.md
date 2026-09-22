# HAM10000 Dataset Notes

## Dataset

- Name: HAM10000, also distributed on Kaggle as Skin Cancer MNIST: HAM10000
- Kaggle page: https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000
- Images: 10,015 dermatoscopic images
- Classes: 7
- Metadata file: `HAM10000_metadata.csv`

## Class codes

| Code | Meaning |
|---|---|
| `akiec` | Actinic keratosis / intraepithelial carcinoma |
| `bcc` | Basal cell carcinoma |
| `bkl` | Benign keratosis-like lesion |
| `df` | Dermatofibroma |
| `mel` | Melanoma |
| `nv` | Melanocytic nevus |
| `vasc` | Vascular lesion |

## Project split

The project uses a stratified 80/20 train-test split with `random_state=42`.

Expected counts:

```text
Training images: 8,012
Test images:    2,003
```

## Imbalance note

The class distribution is heavily skewed toward `nv`. This motivates evaluation beyond accuracy, including class-level recall, macro-average F1 score, weighted-average F1 score, and confusion matrices.

## Data handling

The dataset is not uploaded to this repository. Download it from Kaggle and place the original metadata CSV and image folders in `data/raw/`.

## Medical-use disclaimer

The dataset and code are used for educational machine-learning experimentation. Model outputs are not clinically validated and must not be used for diagnosis, treatment, triage, or medical decision-making.
