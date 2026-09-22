# Data

This project uses the HAM10000 skin-lesion image dataset.

## Required download

Download the dataset from Kaggle:

```text
[https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)
```

## Expected local structure

```text
data/raw/
├── HAM10000_metadata.csv
├── HAM10000_images_part_1/
└── HAM10000_images_part_2/
```

The project expects the original Kaggle filenames and folder names.

## Why data is excluded

The full image dataset is not committed to GitHub because it is large and distributed under separate dataset terms. The root `.gitignore` excludes the contents of `data/raw/` and `data/processed/` while retaining `.gitkeep` files so the folders remain visible.

## Processed data

The scripts construct image paths and encode labels at runtime. If you later save cleaned metadata or split files, place them in `data/processed/`; do not commit them unless they are small, non-sensitive, and permitted for redistribution.
