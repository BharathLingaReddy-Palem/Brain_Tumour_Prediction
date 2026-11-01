"""Exact-pixel deduplication and stratified splits for future training runs.

This cannot infer patient identity or detect all near duplicates. Existing model
weights and the separate evaluation of the original test split are unchanged.
"""
from functools import lru_cache
import hashlib
from pathlib import Path

from PIL import Image
from sklearn.model_selection import train_test_split


@lru_cache(maxsize=50000)
def _pixel_hash(path, size, mtime_ns):
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        return hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest()


def sample_hash(path):
    path = Path(path).resolve()
    stat = path.stat()
    return _pixel_hash(str(path), stat.st_size, stat.st_mtime_ns)


def classification_split_indices(train_samples, test_samples, val_ratio=.15, seed=42):
    """Return unique training/validation/test indices with no exact pixel overlap.

    Reserve the given test set first, exclude its pixels from training, collapse
    repeated training pixels, then stratify the remaining unique samples.
    """
    if not 0 < val_ratio < 1:
        raise ValueError("val_ratio must be strictly between zero and one.")
    known_labels, train_unique, test_unique = {}, {}, {}
    for samples, unique in ((test_samples, test_unique), (train_samples, train_unique)):
        for index, (path, label) in enumerate(samples):
            digest = sample_hash(path)
            if digest in known_labels and known_labels[digest] != label:
                raise ValueError(f"Identical MRI pixels have conflicting class labels: {path}")
            known_labels[digest] = label
            unique.setdefault(digest, index)
    candidates = [idx for digest, idx in train_unique.items() if digest not in test_unique]
    labels = [train_samples[idx][1] for idx in candidates]
    if not candidates:
        raise ValueError("No training images remain after excluding test overlap.")
    train, val = train_test_split(candidates, test_size=val_ratio, random_state=seed, stratify=labels)
    return sorted(train), sorted(val), sorted(test_unique.values())
