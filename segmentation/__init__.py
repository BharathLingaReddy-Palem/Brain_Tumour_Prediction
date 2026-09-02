"""
Brain Tumor Segmentation Module
================================
Provides Graph Cut segmentation for tumor isolation:
  - Normalized Cut (Superpixel RAG)
  - Seeded Min-Cut (Energy Minimization)
  - Adaptive Consensus Ensemble

Usage:
    from segmentation.graph_cut import GraphCutSegmenter
    from segmentation.metrics import SegmentationMetrics
"""
from __future__ import annotations

from segmentation.graph_cut import GraphCutSegmenter
from segmentation.metrics import SegmentationMetrics

__all__ = ["GraphCutSegmenter", "SegmentationMetrics"]
