from .data import DetectorRegionDataset, GuidedQwenDataset, inference_regions, load_detector_cache, make_guided_collator
from .legacy import CATEGORIES, CATEGORY_TO_ID, GuidedConfig, ImplicitGuidanceHead, category_id, hard_select, inject_guides

__all__ = ["DetectorRegionDataset", "GuidedQwenDataset", "inference_regions", "make_guided_collator", "load_detector_cache", "B3Trainer", "CATEGORIES", "CATEGORY_TO_ID", "GuidedConfig", "ImplicitGuidanceHead", "category_id", "hard_select", "inject_guides"]


def __getattr__(name):
    if name == "B3Trainer":
        from .trainer import B3Trainer
        return B3Trainer
    raise AttributeError(name)
