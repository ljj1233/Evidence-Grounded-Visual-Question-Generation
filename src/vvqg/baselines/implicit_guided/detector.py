"""Frozen COCO detector and serializable region-cache helpers for B3.

The detector is a documented substitution for the unavailable historical
bottom-up Faster-RCNN TSV features.  All detector parameters are frozen and
the cache is computed once per unique image.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Sequence

import torch
from torch import nn

# Torchvision's COCO labels use a background entry at index zero.
COCO_CLASSES = (
    "__background__", "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "N/A", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse",
    "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "N/A", "N/A", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "N/A", "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
    "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "N/A", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush",
)


class CocoFasterRCNNRegions(nn.Module):
    """Extract top-scored, padded ROI box-head features from frozen COCO RCNN."""

    def __init__(self, weights: str | Path | None = "DEFAULT", device: str = "cuda", top_k: int = 36):
        super().__init__()
        from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
        from torchvision.models.detection.faster_rcnn import FasterRCNN_ResNet50_FPN_V2_Weights

        if weights is None:
            model = fasterrcnn_resnet50_fpn_v2(weights=None, weights_backbone=None)
            self.weights_id = "torchvision:fasterrcnn_resnet50_fpn_v2:none"
        elif str(weights) == "DEFAULT":
            model = fasterrcnn_resnet50_fpn_v2(weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT)
            self.weights_id = str(FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT.url)
        else:
            model = fasterrcnn_resnet50_fpn_v2(weights=None, weights_backbone=None)
            model.load_state_dict(torch.load(str(weights), map_location="cpu", weights_only=True))
            self.weights_id = str(weights)
        self.model = model.eval().to(device)
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.device, self.top_k = device, int(top_k)
        self.feature_dim = self.model.roi_heads.box_predictor.cls_score.in_features

    @torch.inference_mode()
    def forward(self, images: list[torch.Tensor]) -> list[dict[str, torch.Tensor]]:
        images = [image.to(self.device) for image in images]
        image_batch, _ = self.model.transform(images)
        features = self.model.backbone(image_batch.tensors)
        if isinstance(features, torch.Tensor):
            features = {"0": features}
        proposals, _ = self.model.rpn(image_batch, features, None)
        detections, _ = self.model.roi_heads(features, proposals, image_batch.image_sizes, None)
        result = []
        for image_index, detection in enumerate(detections):
            order = detection["scores"].argsort(descending=True)[: self.top_k]
            boxes = detection["boxes"][order]
            n = boxes.shape[0]
            # box_roi_pool/box_head provides the detector's semantic region
            # representation instead of a string-only object cache.
            if n:
                pooled = self.model.roi_heads.box_roi_pool(
                    {key: value[image_index:image_index + 1] for key, value in features.items()},
                    [boxes], [image_batch.image_sizes[image_index]]
                )
                pooled = self.model.roi_heads.box_head(pooled)
            else:
                pooled = boxes.new_zeros((0, self.feature_dim))
            h, w = image_batch.image_sizes[image_index]
            scale = boxes.new_tensor([w, h, w, h]).clamp_min(1)
            if n < self.top_k:
                boxes = torch.cat((boxes, boxes.new_zeros((self.top_k - n, 4))))
                pooled = torch.cat((pooled, pooled.new_zeros((self.top_k - n, pooled.shape[-1]))))
            labels = detection["labels"][order]
            scores = detection["scores"][order]
            labels = torch.cat((labels, labels.new_zeros(self.top_k - n)))
            scores = torch.cat((scores, scores.new_zeros(self.top_k - n)))
            result.append({
                "features": pooled.float().cpu(),
                "boxes": boxes.float().cpu(),
                "normalized_boxes": (boxes / scale).float().cpu(),
                "class_ids": labels.long().cpu(),
                "labels": labels.long().cpu(),
                "class_labels": [COCO_CLASSES[int(x)] if int(x) < len(COCO_CLASSES) else "N/A" for x in labels],
                "scores": scores.float().cpu(),
                "mask": torch.cat((torch.ones(n, dtype=torch.bool), torch.zeros(self.top_k - n, dtype=torch.bool))),
            })
        return result


def _jsonable(value):
    if torch.is_tensor(value):
        return value.tolist()
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def write_region_cache(path: Path, image_ids: Sequence[str | int], outputs: Sequence[dict]) -> None:
    """Write one JSONL record per image; tensors are converted losslessly to lists."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for image_id, output in zip(image_ids, outputs):
            row = {"image_id": str(image_id), **_jsonable(output)}
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_region_cache(images: Iterable[torch.Tensor], image_ids: Sequence[str | int], detector: CocoFasterRCNNRegions, output: Path | None = None):
    """Run a frozen detector once over a deduplicated image sequence."""
    images = list(images)
    image_ids = [str(item) for item in image_ids]
    if len(image_ids) != len(images):
        raise ValueError("image_ids and images must have equal length")
    rows = detector(list(images))
    cache = {image_id: {"image_id": image_id, **_jsonable(row)} for image_id, row in zip(image_ids, rows)}
    if output is not None:
        write_region_cache(output, image_ids, rows)
    return cache


__all__ = ["COCO_CLASSES", "CocoFasterRCNNRegions", "build_region_cache", "write_region_cache"]
