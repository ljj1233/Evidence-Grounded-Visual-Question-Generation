"""Backward-compatible import path for the B3 detector adapter."""

from .implicit_guided.detector import COCO_CLASSES, CocoFasterRCNNRegions, build_region_cache, write_region_cache

__all__ = ["COCO_CLASSES", "CocoFasterRCNNRegions", "build_region_cache", "write_region_cache"]
