#!/usr/bin/env python3
"""Materialize frozen COCO Faster R-CNN top-36 feature/box caches."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from vvqg.baselines.implicit_guided_detector import CocoFasterRCNNRegions
from vvqg.baselines.common import SEED, write_jsonl
from torchvision.models.detection.faster_rcnn import FasterRCNN_ResNet50_FPN_V2_Weights


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=ROOT / "outputs/same_backbone_baselines/manifests/train50k_images.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/same_backbone_baselines/b3_implicit/cache/detector.jsonl")
    parser.add_argument("--weights", default="DEFAULT")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    from PIL import Image
    import torch
    rows = [json.loads(line) for line in args.manifest.open(encoding="utf-8")]
    rows = rows[:args.limit] if args.limit else rows
    extractor = CocoFasterRCNNRegions(args.weights, args.device)
    categories = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT.meta["categories"] if args.weights == "DEFAULT" else []
    result = []
    for start in range(0, len(rows), 8):
        batch_rows = rows[start:start + 8]
        images = [torch.from_numpy(__import__("numpy").array(Image.open(ROOT / row.get("image_path", row.get("image"))).convert("RGB"))).permute(2, 0, 1).float() / 255 for row in batch_rows]
        features = extractor(images)
        for row, value in zip(batch_rows, features):
            labels = [categories[int(i)] if categories else str(int(i)) for i in value["labels"]]
            labels += ["<empty>"] * (36 - len(labels))
            class_ids = value["class_ids"].tolist()
            class_ids += [-1] * (36 - len(class_ids))
            scores = value["scores"].tolist()
            scores += [0.0] * (36 - len(scores))
            result.append({"image_id": str(row["image_id"]), "candidates": labels[:36], "class_ids": class_ids[:36], "mask": value["mask"].int().tolist(), "features": value["features"].tolist(), "boxes": value["boxes"].tolist(), "normalized_boxes": value["normalized_boxes"].tolist(), "scores": scores[:36], "detector": "torchvision-fasterrcnn-resnet50-fpn-v2-coco", "weights": extractor.weights_id, "seed": SEED})
    write_jsonl(args.output, result)
    print(json.dumps({"images": len(result), "top_k": 36, "output": str(args.output), "weights": extractor.weights_id}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
