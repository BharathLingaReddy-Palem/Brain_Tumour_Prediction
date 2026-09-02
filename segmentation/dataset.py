"""
dataset.py
==========
PyTorch Dataset classes for brain tumor segmentation training.

Supported dataset formats
-------------------------
1. LGG Segmentation (Kaggle / Mateusz Buda)
   Structure:
     dataset_root/
       TCGA_XX_XXXX_XXXXXXXX/
           TCGA_XX_XXXX_XXXXXXXX_<slice>.tif   ← MRI image
           TCGA_XX_XXXX_XXXXXXXX_<slice>_mask.tif  ← Ground truth mask

2. BraTS-style (TCIA / Kaggle BraTS)
   Flat folder with image/mask pairs:
     dataset_root/
       images/  <image>.png
       masks/   <image>_mask.png

3. Generic paired folder
   dataset_root/
     images/  *.jpg / *.png
     masks/   *.jpg / *.png   (same filenames as images)

Usage
-----
    from segmentation.dataset import get_dataloaders
    train_dl, val_dl = get_dataloaders("path/to/lgg-segmentation", fmt="lgg", img_size=256)
"""
from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split


# ---------------------------------------------------------------------------
# Augmentation helpers (pure numpy / PIL, no albumentations dependency)
# ---------------------------------------------------------------------------

def _random_hflip(img: np.ndarray, mask: np.ndarray, p: float = 0.5):
    if random.random() < p:
        img  = np.fliplr(img).copy()
        mask = np.fliplr(mask).copy()
    return img, mask


def _random_vflip(img: np.ndarray, mask: np.ndarray, p: float = 0.3):
    if random.random() < p:
        img  = np.flipud(img).copy()
        mask = np.flipud(mask).copy()
    return img, mask


def _random_rotate90(img: np.ndarray, mask: np.ndarray, p: float = 0.3):
    if random.random() < p:
        k = random.choice([1, 2, 3])
        img  = np.rot90(img,  k, axes=(0, 1)).copy()
        mask = np.rot90(mask, k, axes=(0, 1)).copy()
    return img, mask


def _random_brightness(img: np.ndarray, max_delta: float = 0.15):
    delta = random.uniform(-max_delta, max_delta)
    return np.clip(img + delta, 0.0, 1.0)


def apply_augmentations(img: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Apply random augmentations to (img, mask) pair."""
    img, mask = _random_hflip(img, mask)
    img, mask = _random_vflip(img, mask)
    img, mask = _random_rotate90(img, mask)
    img       = _random_brightness(img)
    return img, mask


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def preprocess_image(pil_img: Image.Image, size: int) -> np.ndarray:
    """
    Convert PIL image → float32 numpy array [0,1], shape (H, W).
    Grayscale — suitable for U-Net input (1 channel).
    """
    img = pil_img.convert("L").resize((size, size), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    # Normalise to zero mean, unit std
    arr = (arr - 0.5) / 0.5
    return arr


def preprocess_mask(pil_mask: Image.Image, size: int) -> np.ndarray:
    """
    Convert PIL mask → binary float32 numpy array [0,1], shape (H, W).
    """
    mask = pil_mask.convert("L").resize((size, size), Image.NEAREST)
    arr  = np.array(mask, dtype=np.float32)
    arr  = (arr > 127).astype(np.float32)   # binarise
    return arr


# ---------------------------------------------------------------------------
# Dataset: LGG Segmentation (Kaggle format)
# ---------------------------------------------------------------------------

def _find_lgg_pairs(root: Path) -> List[Tuple[Path, Path]]:
    """
    Walk the LGG directory tree and collect (image_path, mask_path) pairs.
    Mask files end with '_mask.tif' or '_mask.png'.
    """
    pairs = []
    EXTS = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}

    for patient_dir in sorted(root.iterdir()):
        if not patient_dir.is_dir():
            continue
        files = sorted(patient_dir.glob("*"))
        masks = {
            f.stem.replace("_mask", ""): f
            for f in files
            if "_mask" in f.name and f.suffix.lower() in EXTS
        }
        images = {
            f.stem: f
            for f in files
            if "_mask" not in f.name and f.suffix.lower() in EXTS
        }
        for stem, img_path in images.items():
            if stem in masks:
                pairs.append((img_path, masks[stem]))

    return pairs


class LGGDataset(Dataset):
    """
    Kaggle LGG Brain MRI Segmentation dataset.
    https://www.kaggle.com/mateuszbuda/lgg-segmentation

    Parameters
    ----------
    root      : path to 'lgg-mri-segmentation/kaggle_3m'
    img_size  : target image size (square)
    augment   : apply augmentations (True for train, False for val/test)
    """

    def __init__(self, root: str | Path, img_size: int = 256, augment: bool = False):
        self.root     = Path(root)
        self.img_size = img_size
        self.augment  = augment
        self.pairs    = _find_lgg_pairs(self.root)

        if not self.pairs:
            raise FileNotFoundError(
                f"No LGG image-mask pairs found in {self.root}. "
                "Make sure the dataset is extracted correctly."
            )

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path, mask_path = self.pairs[idx]

        img_arr  = preprocess_image(Image.open(img_path),  self.img_size)
        mask_arr = preprocess_mask(Image.open(mask_path), self.img_size)

        if self.augment:
            img_arr, mask_arr = apply_augmentations(img_arr, mask_arr)

        img_t  = torch.from_numpy(img_arr).unsqueeze(0).float()    # (1, H, W)
        mask_t = torch.from_numpy(mask_arr).unsqueeze(0).float()   # (1, H, W)
        return img_t, mask_t

    def __repr__(self):
        return f"LGGDataset(root={self.root}, pairs={len(self.pairs)}, img_size={self.img_size})"


# ---------------------------------------------------------------------------
# Dataset: Generic paired folder
# ---------------------------------------------------------------------------

class PairedFolderDataset(Dataset):
    """
    Generic dataset where images and masks share the same filename
    but live in separate folders.

    Parameters
    ----------
    images_dir : folder of MRI images
    masks_dir  : folder of corresponding binary masks (same filenames)
    img_size   : target image size
    augment    : apply augmentations
    """

    ALLOWED = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

    def __init__(
        self,
        images_dir: str | Path,
        masks_dir: str | Path,
        img_size: int = 256,
        augment: bool = False,
    ):
        self.images_dir = Path(images_dir)
        self.masks_dir  = Path(masks_dir)
        self.img_size   = img_size
        self.augment    = augment

        self.image_files = sorted(
            p for p in self.images_dir.iterdir()
            if p.is_file() and p.suffix.lower() in self.ALLOWED
        )

        if not self.image_files:
            raise FileNotFoundError(f"No images found in {self.images_dir}")

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path = self.image_files[idx]
        # Try to find matching mask (same stem, any supported extension)
        mask_path = None
        for ext in self.ALLOWED:
            candidate = self.masks_dir / (img_path.stem + ext)
            if candidate.exists():
                mask_path = candidate
                break

        img_arr = preprocess_image(Image.open(img_path), self.img_size)

        if mask_path is not None:
            mask_arr = preprocess_mask(Image.open(mask_path), self.img_size)
        else:
            # No mask available — return zeros (useful for inference-only datasets)
            mask_arr = np.zeros((self.img_size, self.img_size), dtype=np.float32)

        if self.augment:
            img_arr, mask_arr = apply_augmentations(img_arr, mask_arr)

        img_t  = torch.from_numpy(img_arr).unsqueeze(0).float()
        mask_t = torch.from_numpy(mask_arr).unsqueeze(0).float()
        return img_t, mask_t


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def get_dataloaders(
    root: str | Path,
    fmt: str = "lgg",
    img_size: int = 256,
    val_split: float = 0.15,
    test_split: float = 0.05,
    batch_size: int = 16,
    num_workers: int = 4,
    seed: int = 42,
    images_subdir: str = "images",
    masks_subdir: str = "masks",
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build train / val / test DataLoaders.

    Parameters
    ----------
    root       : dataset root directory
    fmt        : 'lgg' for Kaggle LGG format, 'paired' for generic paired folders
    img_size   : image size (square)
    val_split  : fraction of data for validation
    test_split : fraction of data for test
    batch_size : training batch size
    num_workers: DataLoader worker processes
    seed       : random seed for reproducibility
    images_subdir, masks_subdir : subdirectory names (only for 'paired' format)

    Returns
    -------
    train_loader, val_loader, test_loader
    """
    root = Path(root)

    if fmt == "lgg":
        full_ds = LGGDataset(root, img_size=img_size, augment=False)
    elif fmt == "paired":
        full_ds = PairedFolderDataset(
            root / images_subdir,
            root / masks_subdir,
            img_size=img_size,
            augment=False,
        )
    else:
        raise ValueError(f"Unknown dataset format: {fmt!r}. Use 'lgg' or 'paired'.")

    n       = len(full_ds)
    n_val   = int(n * val_split)
    n_test  = int(n * test_split)
    n_train = n - n_val - n_test

    gen = torch.Generator().manual_seed(seed)
    train_ds, val_ds, test_ds = random_split(full_ds, [n_train, n_val, n_test], generator=gen)

    # Enable augmentations on train split only
    if hasattr(train_ds.dataset, "augment"):
        train_ds.dataset.augment = True

    _kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    train_dl = DataLoader(train_ds, shuffle=True,  **_kwargs)
    val_dl   = DataLoader(val_ds,   shuffle=False, **_kwargs)
    test_dl  = DataLoader(test_ds,  shuffle=False, **_kwargs)

    print(
        f"Dataset: {n} samples  |  "
        f"Train: {n_train}  Val: {n_val}  Test: {n_test}  |  "
        f"Format: {fmt}  |  img_size: {img_size}×{img_size}"
    )
    return train_dl, val_dl, test_dl


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    root = Path("Dataset/Testing")
    if not root.exists():
        print("Run from project root. Dataset/Testing not found.")
        sys.exit(0)

    # Test PairedFolderDataset on classification images (no masks)
    ds = PairedFolderDataset(
        images_dir="Dataset/Testing/glioma",
        masks_dir="Dataset/Testing/glioma",   # No mask dir – zeros returned
        img_size=256,
    )
    img, mask = ds[0]
    print(f"Image tensor : {img.shape}  range [{img.min():.2f}, {img.max():.2f}]")
    print(f"Mask tensor  : {mask.shape} range [{mask.min():.2f}, {mask.max():.2f}]")
    print(f"Dataset size : {len(ds)}")
    print("✅ Dataset self-test OK")
