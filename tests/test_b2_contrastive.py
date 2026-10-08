from types import SimpleNamespace

import torch
from torch import nn

from vvqg.baselines.contrastive_grounding.model import ContrastiveGroundingQwen


class TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(32, 8)
        self.lm = nn.Linear(8, 32)
        self.config = SimpleNamespace(text_config=SimpleNamespace(hidden_size=8))

    def forward(self, input_ids, labels, output_hidden_states=False, return_dict=True):
        hidden = self.embed(input_ids)
        logits = self.lm(hidden)
        return SimpleNamespace(
            loss=nn.functional.cross_entropy(logits.flatten(0, 1), labels.flatten()),
            hidden_states=(hidden,),
        )

    def get_input_embeddings(self):
        return self.embed


def test_b2_target_pooling_and_projection_gradients():
    model = ContrastiveGroundingQwen(TinyBackbone(), 5)
    ids = torch.tensor([[1, 2, 3, 4]])
    labels = torch.tensor([[-100, -100, 3, 4]])
    z = nn.functional.normalize(torch.randn(1, 5), dim=-1)
    out = model(input_ids=ids, labels=labels, z_gt=z, z_img=-z, z_txt=z)
    out["loss"].backward()
    assert model.projection.weight.grad is not None
    assert torch.isfinite(out["cl"])


def test_b2_cached_vectors_are_detached(tmp_path):
    from vvqg.baselines.contrastive_grounding.data import load_embedding_cache
    import json

    # The loader explicitly clears autograd even when serialized values are reused.
    path = tmp_path / "embeddings.json"
    path.write_text(json.dumps({"embedding_dim": 2, "vectors": {"x": [1.0, 0.0]}}))
    vectors, _ = load_embedding_cache(path)
    assert vectors["x"].requires_grad is False


def test_b2_duplicate_caption_reuses_text_cache(tmp_path):
    import json
    from vvqg.baselines.contrastive_grounding.data import load_image_caches

    rows = [
        {"image_id": "1", "caption": "same scene", "q_img": "image question"},
        {"image_id": "2", "caption": "same scene", "q_img": "another image question"},
    ]
    (tmp_path / "captions.jsonl").write_text("\n".join(json.dumps({"image_id": r["image_id"], "caption": r["caption"]}) for r in rows) + "\n")
    (tmp_path / "q_img.jsonl").write_text("\n".join(json.dumps({"image_id": r["image_id"], "question": r["q_img"]}) for r in rows) + "\n")
    (tmp_path / "q_txt.jsonl").write_text(json.dumps({"image_id": "1", "caption_hash": __import__("hashlib").sha256(b'same scene').hexdigest(), "question": "text question"}) + "\n")
    cache = load_image_caches(tmp_path)
    assert cache["1"]["q_txt"] == cache["2"]["q_txt"] == "text question"
