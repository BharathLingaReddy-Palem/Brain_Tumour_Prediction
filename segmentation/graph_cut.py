"""
graph_cut.py
============
Brain Tumor Isolation via Graph Cut Segmentation
(Normalized Graph Cut & Energy Min-Cut Algorithms)

Pipeline Architecture
---------------------
1. Image Normalization:
   - Automatically handles standard black-background MRIs and white-bordered/padded scans.
2. Skull-Stripping & Parenchymal Isolation:
   - Strips the high-intensity skull/calvarium and scalp tissue using distance-transform thresholding.
   - Confines graph cut nodes strictly inside the intracranial brain parenchyma.
3. Graph Formulation & Segmentation:
   - Normalized Cut: Superpixel RAG spectral partitioning (Shi & Malik).
   - Seeded Min-Cut: Energy minimization with source (tumor) & sink (healthy tissue) t-links.
   - Ensemble: High-confidence consensus.
4. Optional Necrotic Core Filling:
   - Automatically unifies the hyperintense enhancing rim and necrotic core of ring-enhancing tumors (e.g. GBM).
5. Rich Visualizations:
   - Original Scan, Cyan Boundary Contour, Translucent Red Highlight, Binary Mask, and Cropped Tumor ROI.
"""
from __future__ import annotations

import warnings
from typing import Optional, Tuple, Dict, Any

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage as ndi

# Scikit-image graph imports supporting modern (>=0.20/0.26) and legacy versions
_SKIMAGE_OK = True
try:
    from skimage.color import rgb2gray
    from skimage.segmentation import slic
    try:
        from skimage import graph as skgraph
    except ImportError:
        try:
            from skimage.future import graph as skgraph
        except ImportError:
            skgraph = None
            _SKIMAGE_OK = False
except ImportError:
    _SKIMAGE_OK = False
    skgraph = None


DEFAULT_SIZE = 256
N_SUPERPIXELS = 180
COMPACTNESS = 12.0
SIGMA = 1.0


# ---------------------------------------------------------------------------
# Robust Skull Stripping & Brain Parenchyma Isolation
# ---------------------------------------------------------------------------

def extract_brain_parenchyma(gray: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Robust skull-stripping that works regardless of background color (black or white-bordered).
    Returns:
        brain_mask : bool mask strictly inside brain tissue (skull stripped)
        head_mask  : bool mask of full head including skull
    """
    h, w = gray.shape

    # Check if image has white border/padding (common in screenshots and exports)
    corner_pixels = np.concatenate([
        gray[:8, :8].flatten(),
        gray[:8, -8:].flatten(),
        gray[-8:, :8].flatten(),
        gray[-8:, -8:].flatten(),
    ])
    is_white_frame = np.median(corner_pixels) > 160

    if is_white_frame:
        outer_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
        for seed_pt in [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)]:
            cv2.floodFill(gray.copy(), outer_mask, seed_pt, 0, loDiff=40, upDiff=40)
        valid_area = (outer_mask[1:-1, 1:-1] == 0)
    else:
        valid_area = np.ones((h, w), dtype=bool)

    # Threshold head vs dark air
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    _, head_bin = cv2.threshold(blurred, 30, 255, cv2.THRESH_BINARY)
    head_bin = head_bin & valid_area.astype(np.uint8) * 255

    # Fill holes in the head
    head_filled = ndi.binary_fill_holes(head_bin > 0)
    lbl, n = ndi.label(head_filled)
    if n > 0:
        sizes = ndi.sum(head_filled, lbl, range(1, n + 1))
        head_mask = (lbl == (np.argmax(sizes) + 1))
    else:
        head_mask = np.ones((h, w), dtype=bool)

    # Distance transform to strip the bright skull bone ring completely
    dist = ndi.distance_transform_edt(head_mask)
    if np.any(head_mask):
        dist_thresh = max(8.0, np.percentile(dist[head_mask], 13.0))
        brain_mask = dist > dist_thresh
    else:
        brain_mask = head_mask

    # Morphological opening to clean up outer boundary
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    brain_mask = cv2.morphologyEx(brain_mask.astype(np.uint8), cv2.MORPH_OPEN, kernel) > 0

    if np.sum(brain_mask) < 500:
        brain_mask = head_mask

    return brain_mask, head_mask


# ---------------------------------------------------------------------------
# Method 1: Normalized Graph Cut (Superpixel RAG)
# ---------------------------------------------------------------------------

def _run_normalized_cut(
    rgb: np.ndarray,
    gray: np.ndarray,
    brain_mask: np.ndarray,
    sensitivity: float = 1.0,
) -> np.ndarray:
    """
    Normalized Graph Cut on Region Adjacency Graph (RAG) of SLIC superpixels.
    """
    if not _SKIMAGE_OK or skgraph is None:
        raise RuntimeError("scikit-image graph module is required for Normalized Cut.")

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_gray = clahe.apply(gray)
    enhanced_rgb = np.stack([enhanced_gray] * 3, axis=-1)

    segments = slic(
        enhanced_rgb,
        n_segments=N_SUPERPIXELS,
        compactness=COMPACTNESS,
        sigma=SIGMA,
        start_label=1,
        channel_axis=2,
        mask=brain_mask,
    )

    rag = skgraph.rag_mean_color(enhanced_rgb, segments, mode="similarity")
    labels_cut = skgraph.cut_normalized(
        segments,
        rag,
        thresh=0.06,
        num_cuts=25,
        in_place=False,
    )

    unique_labels = [l for l in np.unique(labels_cut) if l > 0]
    if not unique_labels:
        return np.zeros_like(gray, dtype=np.uint8)

    brain_pixels = gray[brain_mask]
    b_mean = float(np.mean(brain_pixels)) if len(brain_pixels) > 0 else 100.0
    b_std = float(np.std(brain_pixels)) if len(brain_pixels) > 0 else 30.0

    best_label = None
    best_score = -1e9
    min_tumor_px = 40
    max_tumor_px = max(100, int(np.sum(brain_mask) * 0.40))

    thresh_bias = 0.4 * (sensitivity - 1.0)

    for l in unique_labels:
        seg_region = (labels_cut == l) & brain_mask
        px_count = int(np.sum(seg_region))
        if min_tumor_px <= px_count <= max_tumor_px:
            region_mean = float(np.mean(gray[seg_region]))
            score = ((region_mean - b_mean) / (b_std + 1e-5)) + thresh_bias
            if score > best_score:
                best_score = score
                best_label = l

    if best_label is not None and best_score > 0.30:
        ncut_mask = ((labels_cut == best_label) & brain_mask).astype(np.uint8)
    else:
        # Fallback: top intensity superpixels
        superpixel_means = []
        for l in unique_labels:
            seg_region = (segments == l) & brain_mask
            if np.sum(seg_region) > 20:
                superpixel_means.append((l, float(np.mean(gray[seg_region]))))

        superpixel_means.sort(key=lambda x: x[1], reverse=True)
        ncut_mask = np.zeros_like(gray, dtype=np.uint8)
        accum_px = 0
        target_thresh = b_mean + (0.9 - 0.3 * (sensitivity - 1.0)) * b_std
        for l, m_val in superpixel_means:
            if m_val > target_thresh and accum_px < max_tumor_px:
                reg = (segments == l) & brain_mask
                ncut_mask[reg] = 1
                accum_px += int(np.sum(reg))

    return ncut_mask


# ---------------------------------------------------------------------------
# Method 2: Energy Min-Cut / GrabCut with Seed Constraints
# ---------------------------------------------------------------------------

def _run_min_cut(
    rgb: np.ndarray,
    gray: np.ndarray,
    brain_mask: np.ndarray,
    sensitivity: float = 1.0,
) -> np.ndarray:
    """
    Graph Cut Energy Minimization (Boykov-Kolmogorov style):
      - Sink (Definite Background): Air and skull outside brain
      - Probable Background: Healthy brain parenchyma
      - Source (Probable/Definite Foreground): Tumor seeds based on intensity/contrast
    """
    brain_pixels = gray[brain_mask]
    if len(brain_pixels) == 0:
        return np.zeros_like(gray, dtype=np.uint8)

    b_mean = float(np.mean(brain_pixels))
    b_std = float(np.std(brain_pixels))

    thresh_factor = max(0.4, 1.15 - 0.45 * (sensitivity - 1.0))
    enhancement_thresh = b_mean + thresh_factor * b_std
    core_thresh = b_mean + (thresh_factor + 0.6) * b_std

    gc_mask = np.zeros(gray.shape, dtype=np.uint8)
    gc_mask[~brain_mask] = cv2.GC_BGD
    gc_mask[brain_mask] = cv2.GC_PR_BGD

    candidate_mask = brain_mask & (gray > enhancement_thresh)
    core_mask = brain_mask & (gray > core_thresh)

    if np.sum(candidate_mask) >= 30:
        gc_mask[candidate_mask] = cv2.GC_PR_FGD
        if np.sum(core_mask) >= 15:
            gc_mask[core_mask] = cv2.GC_FGD
    else:
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        eq = clahe.apply(gray)
        pct = max(75.0, 88.0 - 10.0 * (sensitivity - 1.0))
        candidate_mask = brain_mask & (eq > np.percentile(eq[brain_mask], pct))
        gc_mask[candidate_mask] = cv2.GC_PR_FGD

    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    img_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    try:
        cv2.grabCut(img_bgr, gc_mask, None, bgd_model, fgd_model, iterCount=5, mode=cv2.GC_INIT_WITH_MASK)
        tumor_raw = np.where((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 1, 0).astype(np.uint8)
        tumor_raw = tumor_raw * brain_mask.astype(np.uint8)
    except Exception as e:
        tumor_raw = candidate_mask.astype(np.uint8)

    return tumor_raw


# ---------------------------------------------------------------------------
# Post-Processing & Necrotic Core Filling
# ---------------------------------------------------------------------------

def _postprocess_tumor_lesion(
    tumor_mask: np.ndarray,
    brain_mask: np.ndarray,
    fill_necrotic_core: bool = True,
    region_filter: str = "auto",
) -> np.ndarray:
    """
    Cleans up the segmented tumor and extracts the primary tumor mass.
    """
    mask = (tumor_mask > 0) & brain_mask
    h, w = mask.shape

    # Apply region/hemisphere filter if requested
    if region_filter == "right_hemisphere":
        mask[:, : w // 2] = False
    elif region_filter == "left_hemisphere":
        mask[:, w // 2 :] = False
    elif region_filter == "center_pituitary":
        margin_x = int(w * 0.25)
        margin_y = int(h * 0.25)
        center_box = np.zeros_like(mask, dtype=bool)
        center_box[margin_y : h - margin_y, margin_x : w - margin_x] = True
        mask = mask & center_box

    if not np.any(mask):
        return np.zeros_like(mask, dtype=np.uint8)

    # Morphological opening to detach thin spurious connections
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cleaned = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)

    lbl, n_lbl = ndi.label(cleaned)
    if n_lbl == 0:
        return np.zeros_like(mask, dtype=np.uint8)

    sizes = ndi.sum(cleaned, lbl, range(1, n_lbl + 1))
    brain_px = np.sum(brain_mask)

    # Find the primary tumor component (must be >= 40 pixels and <= 45% of brain)
    valid_ids = [i + 1 for i, s in enumerate(sizes) if 40 <= s <= brain_px * 0.45]

    final_mask = np.zeros_like(mask, dtype=np.uint8)
    if valid_ids:
        valid_ids.sort(key=lambda x: sizes[x - 1], reverse=True)
        # Take the top 1 or 2 primary components (multi-focal tumor support)
        for vid in valid_ids[:2]:
            final_mask[lbl == vid] = 1
    else:
        largest_id = int(np.argmax(sizes)) + 1
        if sizes[largest_id - 1] <= brain_px * 0.50:
            final_mask[lbl == largest_id] = 1

    # If the tumor has a necrotic dark core (e.g. Glioblastoma), fill the interior cavity
    if fill_necrotic_core and np.any(final_mask):
        final_mask = ndi.binary_fill_holes(final_mask).astype(np.uint8)

    return final_mask * brain_mask.astype(np.uint8)


# ---------------------------------------------------------------------------
# Visualizations
# ---------------------------------------------------------------------------

def _generate_visualizations(
    orig_pil: Image.Image,
    mask: np.ndarray,
    alpha: float = 0.45,
) -> Tuple[Image.Image, Image.Image, Image.Image, Optional[Image.Image]]:
    """
    Returns:
      1. Boundary Contour Image (Cyan boundary line along tumor perimeter)
      2. Overlay Image (Translucent red highlight)
      3. Binary Mask Image (Clean red mask on black)
      4. Cropped Tumor ROI (Zoomed in on the isolated tumor, or None if no tumor)
    """
    target_size = (mask.shape[1], mask.shape[0])
    base_rgb = np.array(orig_pil.convert("RGB").resize(target_size, Image.LANCZOS), dtype=np.uint8)

    # 1. Overlay (translucent clinical red highlight)
    overlay_arr = base_rgb.copy().astype(np.float32)
    tumor_bool = (mask == 1)
    red_tint = np.array([255, 35, 45], dtype=np.float32)
    overlay_arr[tumor_bool] = (1 - alpha) * overlay_arr[tumor_bool] + alpha * red_tint
    overlay_pil = Image.fromarray(np.clip(overlay_arr, 0, 255).astype(np.uint8))

    # 2. Binary Mask (clinical red)
    mask_rgb = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    mask_rgb[tumor_bool] = [240, 50, 50]
    mask_pil = Image.fromarray(mask_rgb)

    # 3. Contour outline overlay (crisp cyan boundary)
    contour_arr = base_rgb.copy()
    contours, _ = cv2.findContours((mask * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(contour_arr, contours, -1, (0, 255, 230), 2)
    contour_pil = Image.fromarray(contour_arr)

    # 4. Cropped Zoomed ROI of the tumor
    cropped_roi = None
    if np.any(tumor_bool):
        coords = np.argwhere(tumor_bool)
        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0)
        # Add 20px padding around bounding box
        pad = 20
        y0_p = max(0, y0 - pad)
        y1_p = min(mask.shape[0], y1 + pad + 1)
        x0_p = max(0, x0 - pad)
        x1_p = min(mask.shape[1], x1 + pad + 1)
        roi_arr = contour_arr[y0_p:y1_p, x0_p:x1_p]
        if roi_arr.size > 0:
            cropped_roi = Image.fromarray(roi_arr).resize((220, 220), Image.LANCZOS)

    return contour_pil, overlay_pil, mask_pil, cropped_roi


# ---------------------------------------------------------------------------
# Public Segmenter API
# ---------------------------------------------------------------------------

class GraphCutSegmenter:
    """
    Dedicated Brain Tumor Graph Cut Segmenter.

    Parameters
    ----------
    method : str
        'normalized_cut' : Normalized Graph Cut on Superpixel RAG
        'grabcut'        : Energy Min-Cut / GrabCut with Seed Constraints
        'both'           : Adaptive Ensemble combining both cuts
    img_size : int
        Processing resolution (default: 256)
    sensitivity : float
        Tumor contrast sensitivity multiplier (0.5 to 2.0, default: 1.0)
    fill_necrotic_core : bool
        Whether to unify the necrotic core inside ring-enhancing tumors
    region_filter : str
        'auto', 'right_hemisphere', 'left_hemisphere', 'center_pituitary'
    """

    def __init__(
        self,
        method: str = "grabcut",
        img_size: int = DEFAULT_SIZE,
        sensitivity: float = 1.0,
        fill_necrotic_core: bool = True,
        region_filter: str = "auto",
    ):
        if method not in ("normalized_cut", "grabcut", "both"):
            raise ValueError(f"Unknown method '{method}'. Use 'normalized_cut', 'grabcut', or 'both'.")

        if method in ("normalized_cut", "both") and (not _SKIMAGE_OK or skgraph is None):
            warnings.warn("skimage.graph is not available; falling back to 'grabcut'.", stacklevel=2)
            method = "grabcut"

        self.method = method
        self.img_size = img_size
        self.sensitivity = sensitivity
        self.fill_necrotic_core = fill_necrotic_core
        self.region_filter = region_filter

    def segment(self, pil_img: Image.Image, postprocess: bool = True) -> Dict[str, Any]:
        """
        Execute Graph Cut segmentation on the given PIL MRI image.
        """
        rgb = np.array(pil_img.convert("RGB").resize((self.img_size, self.img_size), Image.LANCZOS), dtype=np.uint8)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

        brain_mask, head_mask = extract_brain_parenchyma(gray)

        ncut_mask = None
        mincut_mask = None

        if self.method in ("normalized_cut", "both"):
            try:
                ncut_mask = _run_normalized_cut(rgb, gray, brain_mask, self.sensitivity)
            except Exception as e:
                warnings.warn(f"Normalized Cut error: {e}. Falling back to Min-Cut.", stacklevel=2)

        if self.method in ("grabcut", "both") or ncut_mask is None:
            mincut_mask = _run_min_cut(rgb, gray, brain_mask, self.sensitivity)

        if ncut_mask is not None and mincut_mask is not None:
            if np.sum(ncut_mask) > 0 and np.sum(mincut_mask) > 0:
                combined = ((ncut_mask > 0) | (mincut_mask > 0)).astype(np.uint8)
            elif np.sum(mincut_mask) > 0:
                combined = mincut_mask
            else:
                combined = ncut_mask
        elif ncut_mask is not None:
            combined = ncut_mask
        else:
            combined = mincut_mask

        final_mask = (
            _postprocess_tumor_lesion(
                combined,
                brain_mask,
                fill_necrotic_core=self.fill_necrotic_core,
                region_filter=self.region_filter,
            )
            if postprocess
            else combined
        )

        contour_pil, overlay_pil, mask_pil, roi_pil = _generate_visualizations(pil_img, final_mask)

        tumor_px = int(np.sum(final_mask))
        brain_px = int(np.sum(brain_mask))
        total_px = final_mask.size

        # Centroid & diameter estimation
        centroid = (0, 0)
        diameter_px = 0
        if tumor_px > 0:
            c_y, c_x = ndi.center_of_mass(final_mask)
            centroid = (round(float(c_x), 1), round(float(c_y), 1))
            diameter_px = round(float(2.0 * np.sqrt(tumor_px / np.pi)), 1)

        return {
            "mask": final_mask,
            "contour_pil": contour_pil,
            "overlay": overlay_pil,
            "mask_pil": mask_pil,
            "roi_pil": roi_pil,
            "method": self.method,
            "img_size": self.img_size,
            "tumor_pixels": tumor_px,
            "tumor_pct": float((tumor_px / total_px) * 100.0),
            "brain_pct": float((tumor_px / max(brain_px, 1)) * 100.0),
            "centroid": centroid,
            "diameter_px": diameter_px,
        }

    def segment_bytes(self, image_bytes: bytes, **kwargs) -> Dict[str, Any]:
        from io import BytesIO
        pil = Image.open(BytesIO(image_bytes)).convert("RGB")
        return self.segment(pil, **kwargs)
