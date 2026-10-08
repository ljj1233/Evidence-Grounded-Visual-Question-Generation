"""B2 dataset/collator and Trainer loss hook."""

from __future__ import annotations

import math
from pathlib import Path

import torch
from torch.utils.data import Dataset

from vvqg.training.trainer_base import BaselineTrainer, target_token_mask
from vvqg.training.train_sft import resolve_image
from .data import text_hash
from .model import ContrastiveGroundingQwen, make_b2_messages


class B2Dataset(Dataset):
    def __init__(self, processor, records: list[dict], image_cache: dict[str, dict], vectors: dict[str, torch.Tensor], merge_size: int, min_pixels: int = 4096, max_pixels: int = 147456, max_length: int = 512):
        self.processor = processor
        self.records = records
        self.image_cache = image_cache
        self.vectors = vectors
        self.merge_size = merge_size
        self.image_kwargs = {"min_pixels": min_pixels, "max_pixels": max_pixels}; self.max_length = max_length
        for row in records:
            row.setdefault("image_id", Path(row["image"]).stem.rsplit("_", 1)[-1].lstrip("0") or "0")
            cache = image_cache[str(row["image_id"])]
            messages = make_b2_messages(row["question"], cache["caption"])
            row["_full_text"] = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            prompt = processor.apply_chat_template(messages[:1], tokenize=False, add_generation_prompt=True)
            row["_prompt_token_ids"] = processor.tokenizer(prompt, add_special_tokens=False)["input_ids"]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        from PIL import Image
        row = self.records[index]
        with Image.open(resolve_image(row["image"])) as source:
            encoded = self.processor(text=row["_full_text"], images=[source.convert("RGB")], return_tensors="pt", truncation=True, max_length=self.max_length, **self.image_kwargs)
        item = {key: (value if key == "image_grid_thw" else value.squeeze(0)) for key, value in encoded.items() if key in {"input_ids", "attention_mask", "pixel_values", "image_grid_thw", "mm_token_type_ids"}}
        image_expanded = int(encoded["image_grid_thw"].prod().item()) // (self.merge_size * self.merge_size)
        item["prompt_len"] = len(row["_prompt_token_ids"]) + image_expanded - 1
        item["z_gt"] = self.vectors[text_hash(row["question"])].clone()
        item["z_img"] = self.vectors[text_hash(self.image_cache[str(row["image_id"])] ["q_img"])].clone()
        item["z_txt"] = self.vectors[text_hash(self.image_cache[str(row["image_id"])] ["q_txt"])].clone()
        return item


def make_b2_collator(processor):
    def collate(batch):
        padded = processor.tokenizer.pad({"input_ids": [item["input_ids"] for item in batch], "attention_mask": [item["attention_mask"] for item in batch]}, padding=True, return_tensors="pt")
        labels = padded["input_ids"].clone()
        labels[padded["attention_mask"] == 0] = -100
        for index, item in enumerate(batch):
            labels[index, :int(item["prompt_len"])] = -100
        output = {"input_ids": padded["input_ids"], "attention_mask": padded["attention_mask"], "labels": labels, "pixel_values": torch.cat([item["pixel_values"] for item in batch]), "image_grid_thw": torch.cat([item["image_grid_thw"] for item in batch]), "z_gt": torch.stack([item["z_gt"] for item in batch]), "z_img": torch.stack([item["z_img"] for item in batch]), "z_txt": torch.stack([item["z_txt"] for item in batch])}
        if any("mm_token_type_ids" in item for item in batch):
            mm = torch.zeros_like(padded["input_ids"], dtype=batch[0]["mm_token_type_ids"].dtype)
            for index, item in enumerate(batch):
                mm[index, :item["mm_token_type_ids"].shape[0]] = item["mm_token_type_ids"]
            output["mm_token_type_ids"] = mm
        return output
    return collate


class B2Trainer(BaselineTrainer):
    def __init__(self, *args, beta_start: float = 10.0, beta_step: float = 10.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta_start = beta_start
        self.beta_step = beta_step

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        epoch = int(self.state.epoch or 0)
        beta = self.beta_start + self.beta_step * epoch
        outputs = model(beta=beta, **inputs)
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs.loss
        self.log({key: float(value) for key, value in outputs.items() if key != "z_joint" and torch.is_tensor(value) and value.ndim == 0})
        return (loss, outputs) if return_outputs else loss
