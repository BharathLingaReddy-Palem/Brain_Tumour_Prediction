"""
metrics.py
==========
Segmentation evaluation metrics for binary tumor masks.

Metrics implemented
-------------------
  - Dice Score (F1 for segmentation)
  - IoU / Jaccard Index
  - Precision
  - Recall / Sensitivity
  - Specificity
  - Pixel Accuracy
  - Hausdorff Distance (optional, requires scipy)

Usage
-----
    from segmentation.metrics import SegmentationMetrics
    m = SegmentationMetrics(pred_mask, true_mask)
    print(m.summary())
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

try:
    from scipy.ndimage import binary_erosion, distance_transform_edt
    _SCIPY_OK = True
except ImportError:
    _SCIPY_OK = False


# ---------------------------------------------------------------------------
# Core metric functions
# ---------------------------------------------------------------------------

def _flatten_binary(mask: np.ndarray) -> np.ndarray:
    """Convert any mask to a flat bool array."""
    arr = np.asarray(mask)
    if arr.ndim != 2 or arr.size == 0 or not np.isfinite(arr).all():
        raise ValueError("Masks must be finite, non-empty 2D arrays.")
    return arr.astype(bool).ravel()


def _validate_pair(pred, true):
    if np.shape(pred) != np.shape(true):
        raise ValueError(f"Mask shapes must match: {np.shape(pred)} != {np.shape(true)}")
    return _flatten_binary(pred), _flatten_binary(true)


def dice_score(pred: np.ndarray, true: np.ndarray, smooth: float = 1e-6) -> float:
    """
    Dice Similarity Coefficient  =  2·|X∩Y| / (|X| + |Y|)
    Range [0, 1].  1 = perfect overlap.
    """
    p, t = _validate_pair(pred, true)
    intersection = (p & t).sum()
    return float((2.0 * intersection + smooth) / (p.sum() + t.sum() + smooth))


def iou_score(pred: np.ndarray, true: np.ndarray, smooth: float = 1e-6) -> float:
    """
    Intersection over Union (Jaccard Index)  =  |X∩Y| / |X∪Y|
    Range [0, 1].  1 = perfect overlap.
    """
    p, t = _validate_pair(pred, true)
    intersection = (p & t).sum()
    union = (p | t).sum()
    return float((intersection + smooth) / (union + smooth))


def precision_score(pred: np.ndarray, true: np.ndarray, smooth: float = 1e-6) -> float:
    """Precision = TP / (TP + FP)"""
    p, t = _validate_pair(pred, true)
    tp = (p & t).sum()
    fp = (p & ~t).sum()
    return float((tp + smooth) / (tp + fp + smooth))


def recall_score(pred: np.ndarray, true: np.ndarray, smooth: float = 1e-6) -> float:
    """Recall / Sensitivity = TP / (TP + FN)"""
    p, t = _validate_pair(pred, true)
    tp = (p & t).sum()
    fn = (~p & t).sum()
    return float((tp + smooth) / (tp + fn + smooth))


def specificity_score(pred: np.ndarray, true: np.ndarray, smooth: float = 1e-6) -> float:
    """Specificity = TN / (TN + FP)"""
    p, t = _validate_pair(pred, true)
    tn = (~p & ~t).sum()
    fp = (p & ~t).sum()
    return float((tn + smooth) / (tn + fp + smooth))


def pixel_accuracy(pred: np.ndarray, true: np.ndarray) -> float:
    """Pixel Accuracy = (TP + TN) / total pixels"""
    p, t = _validate_pair(pred, true)
    correct = (p == t).sum()
    return float(correct / len(p))


def hausdorff_distance(pred: np.ndarray, true: np.ndarray) -> Optional[float]:
    """
    Hausdorff Distance between the boundary of pred and true masks.
    Returns None if scipy is not available or masks are empty.
    """
    _validate_pair(pred, true)
    if not _SCIPY_OK:
        return None
    p = _flatten_binary(pred).reshape(pred.shape)
    t = _flatten_binary(true).reshape(true.shape)
    if not p.any() or not t.any():
        return None
    p_boundary = p & ~binary_erosion(p, border_value=0)
    t_boundary = t & ~binary_erosion(t, border_value=0)
    d1 = distance_transform_edt(~t_boundary)[p_boundary].max()
    d2 = distance_transform_edt(~p_boundary)[t_boundary].max()
    return float(max(d1, d2))


# ---------------------------------------------------------------------------
# Convenience class
# ---------------------------------------------------------------------------

@dataclass
class SegmentationMetrics:
    """
    Compute and store all segmentation metrics for a single prediction.

    Parameters
    ----------
    pred_mask : np.ndarray  binary predicted mask (0/1 or bool)
    true_mask : np.ndarray  binary ground-truth mask (0/1 or bool)
                            Pass None when ground truth is unavailable —
                            metrics will be set to None.
    method    : str         label for this prediction (e.g. 'Graph Cut')
    """
    pred_mask : np.ndarray
    true_mask : Optional[np.ndarray] = None
    method    : str = "unknown"

    # Computed fields (filled in __post_init__)
    dice        : Optional[float] = field(default=None, init=False)
    iou         : Optional[float] = field(default=None, init=False)
    precision   : Optional[float] = field(default=None, init=False)
    recall      : Optional[float] = field(default=None, init=False)
    specificity : Optional[float] = field(default=None, init=False)
    accuracy    : Optional[float] = field(default=None, init=False)
    hausdorff   : Optional[float] = field(default=None, init=False)
    tumor_pixel_pct : float       = field(default=0.0,  init=False)

    def __post_init__(self):
        _flatten_binary(self.pred_mask)
        if self.true_mask is not None:
            _validate_pair(self.pred_mask, self.true_mask)
        self.tumor_pixel_pct = float(self.pred_mask.astype(bool).mean() * 100.0)

        if self.true_mask is not None:
            self.dice        = dice_score(self.pred_mask, self.true_mask)
            self.iou         = iou_score(self.pred_mask, self.true_mask)
            self.precision   = precision_score(self.pred_mask, self.true_mask)
            self.recall      = recall_score(self.pred_mask, self.true_mask)
            self.specificity = specificity_score(self.pred_mask, self.true_mask)
            self.accuracy    = pixel_accuracy(self.pred_mask, self.true_mask)
            self.hausdorff   = hausdorff_distance(self.pred_mask, self.true_mask)

    # ------------------------------------------------------------------
    def summary(self) -> dict:
        """Return metrics as a plain dict (None → 'N/A' for display)."""
        def _fmt(v):
            return f"{v:.4f}" if v is not None else "N/A"

        return {
            "Method"               : self.method,
            "Dice Score"           : _fmt(self.dice),
            "IoU"                  : _fmt(self.iou),
            "Precision"            : _fmt(self.precision),
            "Recall"               : _fmt(self.recall),
            "Specificity"          : _fmt(self.specificity),
            "Pixel Accuracy"       : _fmt(self.accuracy),
            "Hausdorff Distance"   : _fmt(self.hausdorff),
            "Tumor Region (%)"     : f"{self.tumor_pixel_pct:.2f}%",
        }

    def as_floats(self) -> dict:
        """Return metrics as raw floats (None where unavailable)."""
        return {
            "method"       : self.method,
            "dice"         : self.dice,
            "iou"          : self.iou,
            "precision"    : self.precision,
            "recall"       : self.recall,
            "specificity"  : self.specificity,
            "accuracy"     : self.accuracy,
            "hausdorff"    : self.hausdorff,
            "tumor_pct"    : self.tumor_pixel_pct,
        }

    def __repr__(self):
        rows = "\n  ".join(f"{k}: {v}" for k, v in self.summary().items())
        return f"SegmentationMetrics[\n  {rows}\n]"


# ---------------------------------------------------------------------------
# Multi-method comparison helper
# ---------------------------------------------------------------------------

def compare_methods(*metrics_list: SegmentationMetrics) -> list[dict]:
    """
    Given multiple SegmentationMetrics objects, return a list of summary dicts
    suitable for pd.DataFrame display.

    Example
    -------
    rows = compare_methods(gc_metrics, unet_metrics)
    df   = pd.DataFrame(rows)
    """
    return [m.summary() for m in metrics_list]


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import json

    rng = np.random.default_rng(42)
    h, w = 256, 256

    # Simulate a ground-truth mask (ellipse in centre)
    true = np.zeros((h, w), dtype=np.uint8)
    cv_y, cv_x = h // 2, w // 2
    for y in range(h):
        for x in range(w):
            if ((x - cv_x) / 60) ** 2 + ((y - cv_y) / 40) ** 2 < 1:
                true[y, x] = 1

    # Simulate a slightly imperfect predicted mask
    pred = true.copy()
    noise = rng.integers(0, 2, size=(h, w))
    pred = np.where(rng.random((h, w)) > 0.85, 1 - pred, pred).astype(np.uint8)

    m = SegmentationMetrics(pred_mask=pred, true_mask=true, method="Test Method")
    print(json.dumps(m.summary(), indent=2))
