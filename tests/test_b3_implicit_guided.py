from types import SimpleNamespace

import torch
from torch import nn

from vvqg.baselines.implicit_guided import ImplicitGuidanceHead, hard_select
from vvqg.baselines.implicit_guided_model import GuidedQwenAdapter, register_guide_tokens
from vvqg.baselines.implicit_guided.semantic_alignment import semantic_object_target


def test_padded_regions_never_selected():
    selected = hard_select(torch.tensor([[4.0, 3.0, 100.0]]), k=2, training=False, valid_mask=torch.tensor([[1, 1, 0]]))
    assert torch.equal(selected, torch.tensor([[1.0, 1.0, 0.0]]))


class _Output:
    def __init__(self, loss):
        self.loss = loss


class _TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(32, 8)
        self.lm = nn.Linear(8, 32)
        self.config = SimpleNamespace(hidden_size=8)

    def get_input_embeddings(self):
        return self.embed

    def forward(self, input_ids, labels, output_hidden_states=False):
        logits = self.lm(self.embed(input_ids))
        loss = nn.functional.cross_entropy(logits.flatten(0, 1), labels.flatten())
        return _Output(loss)


def test_question_loss_reaches_selector_and_guide_projection():
    model = GuidedQwenAdapter(_TinyBackbone(), object_dim=4, hidden_dim=8)
    ids = torch.tensor([[1, 2, 3, 4, 5]])
    objects = torch.randn(1, 4, 4)
    result = model(objects, ids, {"vguide_1": 1, "vguide_2": 2, "cguide": 3}, labels=ids)
    result["outputs"].loss.backward()
    assert model.guidance.selector[0].weight.grad.abs().sum() > 0
    assert model.guide_projection.weight.grad.abs().sum() > 0


def test_guidance_losses_have_three_terms():
    head = ImplicitGuidanceHead(4, 8)
    output = head(torch.randn(2, 5, 4), training=True)
    losses = head.losses(output, torch.tensor([1, 2]), torch.zeros(2, 5), torch.tensor(0.5))
    assert set(losses) == {"loss", "question", "category", "object"}
    assert torch.isfinite(losses["loss"])


def test_reserved_tokens_are_registered_once():
    class Tokenizer:
        def __init__(self): self.vocab = {}
        def add_special_tokens(self, payload): self.vocab.update({x: len(self.vocab) for x in payload["additional_special_tokens"]})
        def convert_tokens_to_ids(self, token): return self.vocab[token]
        def __len__(self): return len(self.vocab)
    class Backbone:
        def __init__(self): self.size = None
        def resize_token_embeddings(self, size): self.size = size
    tokenizer, backbone = Tokenizer(), Backbone()
    ids = register_guide_tokens(tokenizer, backbone)
    assert set(ids) == {"vguide_1", "vguide_2", "cguide"}
    assert backbone.size == 3


def test_semantic_alignment_masks_padded_regions():
    def embed(texts):
        return torch.nn.functional.normalize(torch.eye(len(texts), 3)[:len(texts)], dim=-1)
    target = semantic_object_target(["cat", "dog", "<empty>"], "What animal?", "cat", embed)
    assert target[2] == 0.0


def test_inference_regions_include_cached_box_coordinates():
    from vvqg.baselines.implicit_guided.data import inference_regions

    value = {"features": [[1.0, 2.0]], "normalized_boxes": [[0.1, 0.2, 0.3, 0.4]], "mask": [1]}
    result = inference_regions("x", {"x": value})
    assert result["objects"].shape == (1, 1, 6)
    assert torch.equal(result["object_mask"], torch.ones(1, 1, dtype=torch.bool))
