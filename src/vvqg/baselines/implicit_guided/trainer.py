"""B3 forward/loss adapter; optimization is provided by trainer_base."""
from __future__ import annotations
import torch
from vvqg.baselines.implicit_guided_model import GuidedQwenAdapter
from vvqg.training.trainer_base import BaselineTrainer


class B3Trainer(BaselineTrainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        objects = inputs.pop("objects")
        guide_ids = inputs.pop("guide_token_ids")
        object_mask = inputs.pop("object_mask", None)
        category_targets = inputs.pop("category_targets")
        object_targets = inputs.pop("object_targets")
        inputs.pop("image_id", None)
        inputs.pop("boxes", None)
        inputs.pop("labels_text", None)
        inputs.pop("candidate_labels", None)
        result = model(objects=objects, guide_token_ids=guide_ids, object_mask=object_mask, **inputs)
        losses = model.guidance.losses(result["guidance"], category_targets, object_targets, result["outputs"].loss)
        loss = losses["loss"]
        self.log({"loss": float(loss.detach()), "question": float(losses["question"]), "category": float(losses["category"]), "object": float(losses["object"])})
        return (loss, result) if return_outputs else loss


__all__ = ["B3Trainer", "GuidedQwenAdapter"]
