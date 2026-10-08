"""B3 frozen detector cache dataset helpers."""
from __future__ import annotations
import json
from pathlib import Path
import torch
from torch.utils.data import Dataset

from .categories import map_category_records
from vvqg.training.train_sft import resolve_image

GUIDE_TEXT = "<|vguide_1|><|vguide_2|><|cguide|>\nGenerate a question about this image."


class DetectorRegionDataset(Dataset):
    def __init__(self, records: list[dict], detector_cache: dict[str, dict]):
        self.records, self.cache = records, detector_cache
    def __len__(self): return len(self.records)
    def __getitem__(self, index):
        row = self.records[index]; value = self.cache[str(row["image_id"])]
        output = {
            "image_id": str(row["image_id"]),
            "objects": torch.cat((torch.tensor(value["features"], dtype=torch.float32), torch.tensor(value.get("normalized_boxes", value["boxes"]), dtype=torch.float32)), dim=-1),
            "boxes": torch.tensor(value["boxes"], dtype=torch.float32),
            "object_mask": torch.tensor(value.get("mask", [1] * len(value["features"])), dtype=torch.bool),
            "labels": value.get("class_labels", value.get("candidates", [])),
        }
        if row.get("category_id") is not None:
            output["category_targets"] = torch.tensor(row["category_id"], dtype=torch.long)
        if "object_target" in row:
            output["object_targets"] = torch.tensor(row["object_target"], dtype=torch.float32)
        return output


def load_detector_cache(path: Path) -> dict[str, dict]:
    return {str((row := json.loads(line))["image_id"]): row for line in path.open(encoding="utf-8") if line.strip()}


def attach_category_targets(records: list[dict], mapping: dict[str, tuple[str, ...]] | None = None) -> tuple[list[dict], dict]:
    """Attach official category IDs and return the required coverage report."""
    return map_category_records(records, mapping=mapping)


class GuidedQwenDataset(Dataset):
    """Qwen inputs paired with detector regions and train-only guide targets."""
    def __init__(self, processor, records: list[dict], detector_cache: dict[str, dict], merge_size: int, min_pixels: int = 4096, max_pixels: int = 147456, max_length: int = 512):
        self.processor, self.records, self.cache, self.merge_size = processor, records, detector_cache, merge_size
        self.image_kwargs = {"min_pixels": min_pixels, "max_pixels": max_pixels}; self.max_length = max_length
        for row in records:
            messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": GUIDE_TEXT}]}, {"role": "assistant", "content": row["question"]}]
            row["_full_text"] = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            prompt = processor.apply_chat_template(messages[:1], tokenize=False, add_generation_prompt=True)
            row["_prompt_token_ids"] = processor.tokenizer(prompt, add_special_tokens=False)["input_ids"]

    def __len__(self): return len(self.records)

    def __getitem__(self, index):
        from PIL import Image
        row = self.records[index]
        with Image.open(resolve_image(row["image"])) as source:
            encoded = self.processor(text=row["_full_text"], images=[source.convert("RGB")], return_tensors="pt", truncation=True, max_length=self.max_length, **self.image_kwargs)
        item = {key: (value if key == "image_grid_thw" else value.squeeze(0)) for key, value in encoded.items() if key in {"input_ids", "attention_mask", "pixel_values", "image_grid_thw", "mm_token_type_ids"}}
        item["prompt_len"] = len(row["_prompt_token_ids"]) + int(encoded["image_grid_thw"].prod().item()) // (self.merge_size * self.merge_size) - 1
        value = self.cache[str(row["image_id"])]
        features = torch.tensor(value["features"], dtype=torch.float32)
        boxes = torch.tensor(value.get("normalized_boxes", value["boxes"]), dtype=torch.float32)
        item["objects"] = torch.cat((features, boxes), dim=-1)
        item["object_mask"] = torch.tensor(value.get("mask", [1] * len(value["features"])), dtype=torch.bool)
        item["category_targets"] = torch.tensor(row["category_id"], dtype=torch.long)
        item["object_targets"] = torch.tensor(row["object_target"], dtype=torch.float32)
        return item


def make_guided_collator(processor, guide_token_ids):
    def collate(batch):
        padded = processor.tokenizer.pad({"input_ids": [x["input_ids"] for x in batch], "attention_mask": [x["attention_mask"] for x in batch]}, padding=True, return_tensors="pt")
        labels = padded["input_ids"].clone(); labels[padded["attention_mask"] == 0] = -100
        for i, item in enumerate(batch): labels[i, :int(item["prompt_len"])] = -100
        output = {"input_ids": padded["input_ids"], "attention_mask": padded["attention_mask"], "labels": labels, "pixel_values": torch.cat([x["pixel_values"] for x in batch]), "image_grid_thw": torch.cat([x["image_grid_thw"] for x in batch]), "objects": torch.stack([x["objects"] for x in batch]), "object_mask": torch.stack([x["object_mask"] for x in batch]), "category_targets": torch.stack([x["category_targets"] for x in batch]), "object_targets": torch.stack([x["object_targets"] for x in batch]), "guide_token_ids": guide_token_ids}
        if any("mm_token_type_ids" in x for x in batch):
            mm = torch.zeros_like(padded["input_ids"], dtype=batch[0]["mm_token_type_ids"].dtype)
            for i, item in enumerate(batch): mm[i, :item["mm_token_type_ids"].shape[0]] = item["mm_token_type_ids"]
            output["mm_token_type_ids"] = mm
        return output
    return collate


def inference_regions(image_id: str | int, detector_cache: dict[str, dict]) -> dict[str, torch.Tensor]:
    """Build image-only selector inputs; no category or answer fields are read."""
    value = detector_cache[str(image_id)]
    features = torch.tensor(value["features"], dtype=torch.float32)
    boxes = torch.tensor(value["normalized_boxes"] if "normalized_boxes" in value else value["boxes"], dtype=torch.float32)
    return {
        "objects": torch.cat((features, boxes), dim=-1).unsqueeze(0),
        "object_mask": torch.tensor(value.get("mask", [1] * len(value["features"])), dtype=torch.bool).unsqueeze(0),
    }
