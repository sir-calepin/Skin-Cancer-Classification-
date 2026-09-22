from pathlib import Path

import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision import transforms

from src.config import (
    BATCH_SIZE,
    CLASS_NAMES,
    IMAGE_DIR_PART_1,
    IMAGE_DIR_PART_2,
    IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    METADATA_PATH,
    RANDOM_STATE,
)


def resolve_image_path(image_id: str) -> Path:
    path_part_1 = IMAGE_DIR_PART_1 / f"{image_id}.jpg"
    path_part_2 = IMAGE_DIR_PART_2 / f"{image_id}.jpg"

    if path_part_1.exists():
        return path_part_1
    if path_part_2.exists():
        return path_part_2

    raise FileNotFoundError(
        f"Could not find image '{image_id}.jpg' in:\n"
        f"  {IMAGE_DIR_PART_1}\n"
        f"  {IMAGE_DIR_PART_2}"
    )


def load_metadata(metadata_path: Path = METADATA_PATH) -> pd.DataFrame:
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Metadata file not found: {metadata_path}\n"
            "Download HAM10000 and place it in data/raw/. "
            "See data/README.md for the expected structure."
        )

    dataframe = pd.read_csv(metadata_path)

    required_columns = {"image_id", "dx"}
    missing_columns = required_columns.difference(dataframe.columns)
    if missing_columns:
        raise ValueError(
            f"Metadata is missing required columns: {sorted(missing_columns)}"
        )

    dataframe = dataframe.copy()
    dataframe["label"] = dataframe["dx"].map(
        {class_name: index for index, class_name in enumerate(CLASS_NAMES)}
    )

    if dataframe["label"].isna().any():
        unknown_labels = dataframe.loc[dataframe["label"].isna(), "dx"].unique()
        raise ValueError(f"Unexpected labels found: {unknown_labels}")

    dataframe["label"] = dataframe["label"].astype(int)
    dataframe["path"] = dataframe["image_id"].apply(resolve_image_path)
    return dataframe


def create_train_test_split(
    dataframe: pd.DataFrame,
    test_size: float = 0.20,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_dataframe, test_dataframe = train_test_split(
        dataframe,
        test_size=test_size,
        stratify=dataframe["label"],
        random_state=random_state,
    )

    return (
        train_dataframe.reset_index(drop=True),
        test_dataframe.reset_index(drop=True),
    )


def get_train_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def get_test_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


class HAM10000Dataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, transform=None) -> None:
        self.dataframe = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, index: int):
        row = self.dataframe.iloc[index]
        image = Image.open(row["path"]).convert("RGB")
        label = int(row["label"])

        if self.transform is not None:
            image = self.transform(image)

        return image, label


def create_datasets_and_loaders(
    batch_size: int = BATCH_SIZE,
    num_workers: int = 0,
):
    dataframe = load_metadata()
    train_dataframe, test_dataframe = create_train_test_split(dataframe)

    train_dataset = HAM10000Dataset(
        train_dataframe,
        transform=get_train_transform(),
    )
    test_dataset = HAM10000Dataset(
        test_dataframe,
        transform=get_test_transform(),
    )

    from torch.utils.data import DataLoader

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_dataframe, test_dataframe, train_loader, test_loader
