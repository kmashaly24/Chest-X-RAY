"""Data download, preprocessing, Dataset and DataLoader utilities."""
import io
import os
import zipfile

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

import config


# directories inside data.zip that contain the PNG images
IMAGE_DIR_PREFIXES = [f"images_{i:03d}/images/" for i in range(1, 13)]


def download_data(download_path, dataset="nih-chest-xrays/data"):
    """Download the dataset zip from Kaggle (requires Kaggle credentials).

    Credentials are read from the KAGGLE_USERNAME / KAGGLE_KEY environment
    variables or the default ~/.kaggle/kaggle.json file.
    """
    import kaggle
    os.makedirs(download_path, exist_ok=True)
    kaggle.api.dataset_download_files(dataset, path=download_path, unzip=False)
    return os.path.join(download_path, "data.zip")


def extract_csv(zip_path, download_path):
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(path=download_path, members=["Data_Entry_2017.csv"])
    return os.path.join(download_path, "Data_Entry_2017.csv")


def build_dataframe(csv_path, zip_path):
    """Load the metadata CSV and map every image index to its archive path."""
    df = pd.read_csv(csv_path)
    image_to_path = {}
    with zipfile.ZipFile(zip_path, "r") as zf:
        for name in zf.namelist():
            if name.endswith(".png") and any(name.startswith(p) for p in IMAGE_DIR_PREFIXES):
                image_to_path[os.path.basename(name)] = name
    df["Image Path"] = df["Image Index"].map(image_to_path)
    df = df[["Finding Labels", "Patient ID", "Image Path"]]
    return df


def patient_split(df, seed=42):
    """Patient-level 70/15/15 split (no leakage between splits)."""
    unique_patients = df["Patient ID"].unique()
    train_patients, temp_patients = train_test_split(unique_patients, test_size=0.30, random_state=seed)
    val_patients, test_patients = train_test_split(temp_patients, test_size=0.50, random_state=seed)

    train_df = df[df["Patient ID"].isin(train_patients)].reset_index(drop=True)
    val_df = df[df["Patient ID"].isin(val_patients)].reset_index(drop=True)
    test_df = df[df["Patient ID"].isin(test_patients)].reset_index(drop=True)
    return train_df, val_df, test_df


def worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is not None:
        worker_info.dataset._zipfile = zipfile.ZipFile(worker_info.dataset.data_zip_path, "r")


class SmartChestDatasetV2(Dataset):
    """Stream chest X-rays and multi-hot labels straight from data.zip."""

    def __init__(self, dataframe, download_path, transform=None, label_list=None):
        self.df = dataframe.reset_index(drop=True)
        self.download_path = download_path
        self.transform = transform
        self.labels = label_list
        self.data_zip_path = os.path.join(download_path, "data.zip")
        self._zipfile = None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if self._zipfile is None:
            self._zipfile = zipfile.ZipFile(self.data_zip_path, "r")
        path = self.df.loc[idx, "Image Path"]
        try:
            with self._zipfile.open(path) as f:
                image = Image.open(f).convert("RGB")
        except Exception:
            image = Image.new("RGB", (config.IMG_SIZE, config.IMG_SIZE), (0, 0, 0))

        label_vector = torch.zeros(len(self.labels), dtype=torch.float32)
        finding_labels = self.df.loc[idx, "Finding Labels"]
        for i, label in enumerate(self.labels):
            if label in finding_labels:
                label_vector[i] = 1.0

        image = self.transform(image) if self.transform else transforms.ToTensor()(image)
        return image, label_vector


def build_loaders(train_df, val_df, test_df, download_path, batch_size, workers=0):
    train_dataset = SmartChestDatasetV2(train_df, download_path, transform=config.train_transform, label_list=config.ALL_LABELS)
    val_dataset = SmartChestDatasetV2(val_df, download_path, transform=config.eval_transform, label_list=config.ALL_LABELS)
    test_dataset = SmartChestDatasetV2(test_df, download_path, transform=config.eval_transform, label_list=config.ALL_LABELS)

    common = dict(num_workers=workers, pin_memory=torch.cuda.is_available(), persistent_workers=(workers > 0))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, worker_init_fn=worker_init_fn, drop_last=True, **common)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, worker_init_fn=worker_init_fn, **common)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, worker_init_fn=worker_init_fn, **common)
    return train_loader, val_loader, test_loader