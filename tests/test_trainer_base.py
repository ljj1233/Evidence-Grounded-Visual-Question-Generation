from __future__ import annotations

import torch
from torch import nn

from vvqg.training.trainer_base import BaselineTrainer, target_token_mask


def test_target_mask_matches_plain_sft_labels():
    labels = torch.tensor([[-100, 4, 5, -100], [2, -100, 3, 1]])
    assert torch.equal(target_token_mask(labels), labels.ne(-100))


def test_plain_sft_forward_loss_regression_matches_shared_hook():
    class Tiny(nn.Module):
        def forward(self, input_ids, labels):
            logits = torch.tensor(
                [[[2.0, 0.0, -1.0], [0.5, 1.5, -0.5], [1.0, -1.0, 2.0]]],
                dtype=torch.float32,
                requires_grad=True,
            )
            return {"loss": nn.functional.cross_entropy(
                logits.reshape(-1, 3), labels.reshape(-1), ignore_index=-100
            )}

    class Args:
        output_dir = "."

    # The shared hook is intentionally formula-identical to Trainer's plain CE path.
    hook = BaselineTrainer.compute_loss
    inputs = {"input_ids": torch.tensor([[0, 1, 2]]), "labels": torch.tensor([[-100, 1, 2]])}
    got = hook(None, Tiny(), inputs)
    expected = nn.functional.cross_entropy(
        torch.tensor([[2.0, 0.0, -1.0], [0.5, 1.5, -0.5], [1.0, -1.0, 2.0]]),
        inputs["labels"].reshape(-1), ignore_index=-100,
    )
    assert abs(float(got - expected)) < 1e-6
