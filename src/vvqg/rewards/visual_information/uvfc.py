"""Online mirror of the frozen Phase-2.6 UVFC feature pipeline.

Feature extraction, the Top-K SAE and the UVFC formulation are the frozen
implementation (`scripts/vvqg/phase26_vfc.py`, `docs/method/UVFC_IMPLEMENTATION_FREEZE.md`).
Nothing here redefines them; it only packages the exact same math so a trainer
can score fresh rollouts with one loaded model.  `test_online_uvfc.py` asserts
numerical equality against the frozen `scores4000.jsonl`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import torch

LAYER = 18
HIDDEN = 4096
DICT = 4096
TOPK = 20
TOKEN_CAP = 16
EPS = 1e-6
INSTRUCTION = "Generate a question about this image."
REPO_ROOT = Path(__file__).resolve().parents[4]


class TopKSAE(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty(HIDDEN, DICT))
        self.bias = torch.nn.Parameter(torch.zeros(HIDDEN))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = torch.relu((x - self.bias) @ self.weight)
        vals, idx = torch.topk(z, TOPK, dim=-1)
        sparse = torch.zeros_like(z).scatter(1, idx, vals)
        return sparse, sparse @ self.weight.T + self.bias


def load_uvfc(vfc_dir: str | Path, device: Any) -> dict[str, Any]:
    """Load the frozen SAE and the frozen v/u weights."""
    vfc_dir = Path(vfc_dir)
    sae = TopKSAE()
    sae.load_state_dict(torch.load(vfc_dir / "sae.pt", map_location="cpu")["state_dict"])
    sae = sae.to(device).eval().requires_grad_(False)
    weights = torch.load(vfc_dir / "weights.pt", map_location="cpu")
    return {
        "sae": sae,
        "v": weights["v"].to(device).float(),
        "u": weights["u"].to(device).float(),
        "p_anchor": weights["p_anchor"].to(device).float(),
        "p_policy_frozen": weights["p_policy"].to(device).float(),
        "eps": float(weights.get("eps", EPS)),
    }


def _messages(processor: Any, question: str) -> tuple[str, list[int]]:
    messages = [
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": INSTRUCTION}]},
        {"role": "assistant", "content": question},
    ]
    full = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return full, processor.tokenizer.encode(question, add_special_tokens=False)


def load_image(path: str):
    from PIL import Image

    candidate = Path(path)
    with Image.open(candidate if candidate.is_absolute() else REPO_ROOT / candidate) as source:
        return source.convert("RGB")


@torch.inference_mode()
def _question_pooled_sae(
    model: Any, processor: Any, rows: Sequence[dict[str, Any]], sae: TopKSAE, device: Any
) -> tuple[torch.Tensor, list[int]]:
    """Return (pooled SAE activations [n, DICT], question token counts)."""
    images, texts, qids = [], [], []
    for row in rows:
        images.append(load_image(row["image"]))
        text, ids = _messages(processor, row["question"])
        texts.append(text)
        qids.append(ids)
    encoded = processor(text=texts, images=images, padding=True, return_tensors="pt")
    grids = encoded["image_grid_thw"]
    batch = {k: v.to(device) if hasattr(v, "to") else v for k, v in encoded.items()}
    hidden = model(**batch, use_cache=False, output_hidden_states=True).hidden_states[LAYER + 1]
    vectors, counts = [], []
    for i, ids in enumerate(qids):
        n = int(encoded["attention_mask"][i].sum())
        seq = encoded["input_ids"][i, :n].tolist()
        found = next((j for j in range(len(seq) - len(ids) + 1) if seq[j : j + len(ids)] == ids), None)
        if found is None:
            raise RuntimeError("question token alignment mismatch")
        positions = torch.arange(found, min(found + TOKEN_CAP, found + len(ids)), device=device)
        vectors.append(hidden[i, positions].float())
        counts.append(len(ids))
    del hidden, batch
    padded = torch.nn.utils.rnn.pad_sequence(vectors, batch_first=True)[:, :TOKEN_CAP]
    n, token_cap, hidden_size = padded.shape
    z, _ = sae(padded.reshape(-1, hidden_size).to(device=device, dtype=torch.float32))
    z = z.view(n, token_cap, DICT)
    length_t = torch.tensor([min(c, TOKEN_CAP) for c in counts], device=device).clamp_min(1)
    mask = torch.arange(token_cap, device=device)[None, :] < length_t[:, None]
    pooled = z.masked_fill(~mask[:, :, None], 0).amax(dim=1)
    for image in images:
        image.close()
    return pooled, counts


@torch.inference_mode()
def score_uvfc(
    model: Any, processor: Any, rows: Sequence[dict[str, Any]], frozen: dict[str, Any], device: Any,
    u_override: torch.Tensor | None = None,
) -> list[dict[str, Any]]:
    """Attach ``UVFC``, ``VFC`` and ``active_features`` to each row (does not mutate)."""
    sae, v = frozen["sae"], frozen["v"]
    u = frozen["u"] if u_override is None else u_override.to(device).float()
    pooled, counts = _question_pooled_sae(model, processor, rows, sae, device)
    active = pooled > 0
    vfc = (active * v[None, :]).sum(dim=1)
    uvfc = (active * (v * u)[None, :]).sum(dim=1)
    active_cpu = active.cpu()
    return [
        {
            "UVFC": float(uvfc[i]),
            "VFC": float(vfc[i]),
            "question_token_count": int(counts[i]),
            "active_features": torch.nonzero(active_cpu[i]).squeeze(1).tolist(),
            "_active": active_cpu[i],
        }
        for i in range(len(rows))
    ]


def dynamic_u(p_anchor: torch.Tensor, active_counts: torch.Tensor, total: int, eps: float = EPS) -> torch.Tensor:
    """Fresh undercoverage from the current-policy active-feature distribution."""
    p_policy = active_counts.float() / max(int(total), 1)
    return torch.log((p_anchor + eps) / (p_policy + eps)).clamp_min(0)
