"""Severstal Steel Defect Detection icin PyTorch Dataset.

train.csv kolonlari: ImageId, ClassId (1-4), EncodedPixels (RLE).
Her goruntu icin 4 kanalli (defect turu basina 1) multilabel maske uretilir.
"""
import os

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .rle import rle_decode

NUM_CLASSES = 4
IMG_SHAPE = (256, 1600)  # (H, W) — Severstal goruntulerinin sabit boyutu


class SeverstalDataset(Dataset):
    def __init__(self, csv_path: str, images_dir: str, image_ids=None, transforms=None):
        df = pd.read_csv(csv_path)
        df = df.dropna(subset=["EncodedPixels"])  # defect'i olmayan (bos) satirlari at
        self.images_dir = images_dir
        self.transforms = transforms

        # hizli erisim icin: image_id -> {class_id: rle}
        self.rle_by_image: dict[str, dict[int, str]] = {}
        for row in df.itertuples(index=False):
            self.rle_by_image.setdefault(row.ImageId, {})[int(row.ClassId)] = row.EncodedPixels

        if image_ids is not None:
            # train/val split disaridan verilirse sadece o alt kumeyi kullan
            self.image_ids = list(image_ids)
        else:
            self.image_ids = sorted(self.rle_by_image.keys())

    def __len__(self) -> int:
        return len(self.image_ids)

    def _build_mask(self, image_id: str) -> np.ndarray:
        # (H, W, 4) multilabel maske: her kanal bagimsiz bir defect sinifi (ortusme mumkun)
        mask = np.zeros((*IMG_SHAPE, NUM_CLASSES), dtype=np.float32)
        for class_id, rle in self.rle_by_image.get(image_id, {}).items():
            mask[:, :, class_id - 1] = rle_decode(rle, IMG_SHAPE)
        return mask

    def __getitem__(self, idx: int):
        image_id = self.image_ids[idx]
        image_path = os.path.join(self.images_dir, image_id)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # cv2 BGR okur, model RGB bekler
        mask = self._build_mask(image_id)

        if self.transforms is not None:
            # Albumentations image+mask'i birlikte augment eder (crop/flip vs. ikisine de ayni sekilde uygulanir)
            augmented = self.transforms(image=image, mask=mask)
            image, mask = augmented["image"], augmented["mask"]
            mask = mask.permute(2, 0, 1).float() if torch.is_tensor(mask) else torch.from_numpy(mask).permute(2, 0, 1).float()
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float()
            mask = torch.from_numpy(mask).permute(2, 0, 1).float()

        return image, mask  # image: (3,H,W), mask: (4,H,W)


def train_val_split(csv_path: str, val_size: float = 0.15, seed: int = 42):
    from sklearn.model_selection import train_test_split

    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["EncodedPixels"])
    all_ids = sorted(df["ImageId"].unique())
    # image_id bazinda split -> ayni goruntunun farkli defect siniflari train/val arasinda bolunmesin
    train_ids, val_ids = train_test_split(all_ids, test_size=val_size, random_state=seed)
    return train_ids, val_ids
