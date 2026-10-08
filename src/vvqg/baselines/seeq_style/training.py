"""SeeQ dual-format dataset and round-training helpers."""
from __future__ import annotations
from pathlib import Path
import json
import torch
from torch.utils.data import Dataset
from vvqg.training.train_sft import resolve_image
from vvqg.training.trainer_base import PlainSFTTrainer


def build_messages(row: dict) -> list[dict]:
    if row["format"] == "qg":
        user = [{"type": "image"}, {"type": "text", "text": "Generate a question about this image."}]
        target = row["question"] + " " + row["answer"]
    elif row["format"] == "qa":
        user = [{"type": "image"}, {"type": "text", "text": "Answer this question about the image: " + row["question"]}]
        target = row["answer"]
    else:
        raise ValueError(f"unknown SeeQ format: {row['format']}")
    return [{"role": "user", "content": user}, {"role": "assistant", "content": target}]


class SeeQDataset(Dataset):
    def __init__(self, processor, records: list[dict], merge_size: int, min_pixels: int = 4096, max_pixels: int = 147456, max_length: int = 512, message_builder=None):
        self.processor, self.records, self.merge_size = processor, records, merge_size
        self.image_kwargs = {"min_pixels": min_pixels, "max_pixels": max_pixels}; self.max_length = max_length
        self.message_builder = message_builder or build_messages
        for row in records:
            messages = self.message_builder(row)
            row["_full_text"] = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            row["_prompt_token_ids"] = processor.tokenizer(processor.apply_chat_template(messages[:1], tokenize=False, add_generation_prompt=True), add_special_tokens=False)["input_ids"]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        from PIL import Image
        row = self.records[index]
        with Image.open(resolve_image(row["image"])) as source:
            encoded = self.processor(text=row["_full_text"], images=[source.convert("RGB")], return_tensors="pt", truncation=True, max_length=self.max_length, **self.image_kwargs)
        item = {k: (v if k == "image_grid_thw" else v.squeeze(0)) for k, v in encoded.items() if k in {"input_ids", "attention_mask", "pixel_values", "image_grid_thw", "mm_token_type_ids"}}
        item["prompt_len"] = len(row["_prompt_token_ids"]) + int(encoded["image_grid_thw"].prod().item()) // (self.merge_size * self.merge_size) - 1
        return item


def make_collator(processor):
    def collate(batch):
        padded = processor.tokenizer.pad({"input_ids": [x["input_ids"] for x in batch], "attention_mask": [x["attention_mask"] for x in batch]}, padding=True, return_tensors="pt")
        labels = padded["input_ids"].clone(); labels[padded["attention_mask"] == 0] = -100
        for i, item in enumerate(batch): labels[i, :int(item["prompt_len"])] = -100
        out = {"input_ids": padded["input_ids"], "attention_mask": padded["attention_mask"], "labels": labels, "pixel_values": torch.cat([x["pixel_values"] for x in batch]), "image_grid_thw": torch.cat([x["image_grid_thw"] for x in batch])}
        if any("mm_token_type_ids" in x for x in batch):
            mm = torch.zeros_like(padded["input_ids"], dtype=batch[0]["mm_token_type_ids"].dtype)
            for i, item in enumerate(batch): mm[i, :item["mm_token_type_ids"].shape[0]] = item["mm_token_type_ids"]
            out["mm_token_type_ids"] = mm
        return out
    return collate


SeeQTrainer = PlainSFTTrainer

def round_manifest(round_dir: Path, m0_hash: str, records: int) -> Path:
    payload = {"m0_hash": m0_hash, "records": records, "status": "ready_for_training"}
    path = round_dir / "training_manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path
