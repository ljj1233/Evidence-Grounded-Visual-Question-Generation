#!/usr/bin/env python3
"""Phase 2.6 Track B: train-only FAC-inspired visual feature coverage.

The script is deliberately split into resumable stages.  SAE discovery and all
feature-frequency calibration read only the 50k SFT train manifest; candidate
scoring is a separate stage and never changes the SAE or its weights.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

import torch
from tqdm.auto import tqdm
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from vvqg.paths import local_model_path
BASE = Path(local_model_path('Qwen3-VL-8B-Instruct'))
ADAPTER = ROOT / "outputs/checkpoints/qwen3vl8b_lora_r16_vqav2_50k/checkpoint-final"
TRAIN = ROOT / "src/vvqg/data/sft/train.jsonl"
CANDIDATES = ROOT / "outputs/reward_calibration/sft_candidates_4x.jsonl"
CANON = ROOT / "outputs/phase25_offline_reward_audit/candidates4000.jsonl"
OUT = ROOT / "outputs/phase26/vfc"
SEED = 42
INSTRUCTION = "Generate a question about this image."
N_DISCOVERY = 20_000
N_CALIBRATION = 5_000
TOKEN_CAP = 16
LAYER = None
HIDDEN = None
DICT = 4096  # fixed feasible paper-inspired configuration, chosen before dev scoring
TOPK = 20
POLICY_MANIFEST = None  # optional explicit policy image manifest (else TRAIN unique images)


def configure_model() -> None:
    global LAYER, HIDDEN
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(str(BASE), local_files_only=True)
    text_config = getattr(config, "text_config", config)
    LAYER = int(text_config.num_hidden_layers) // 2
    HIDDEN = int(text_config.hidden_size)


def read(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


def pad_token_axis(x: torch.Tensor) -> torch.Tensor:
    if x.shape[1] == TOKEN_CAP:
        return x
    if x.shape[1] > TOKEN_CAP:
        raise RuntimeError("activation token dimension exceeds TOKEN_CAP")
    out = torch.zeros((x.shape[0], TOKEN_CAP, x.shape[2]), dtype=x.dtype)
    out[:, :x.shape[1]] = x
    return out


def train_split() -> tuple[list[dict], list[dict]]:
    rows = read(TRAIN)
    if len(rows) != 50_000:
        raise RuntimeError(f"expected 50,000 SFT train rows, found {len(rows)}")
    rng = random.Random(SEED)
    order = list(range(len(rows)))
    rng.shuffle(order)
    discovery = [rows[i] for i in order[:N_DISCOVERY]]
    calibration = [rows[i] for i in order[N_DISCOVERY:N_DISCOVERY + N_CALIBRATION]]
    return discovery, calibration


def manifest() -> None:
    discovery, calibration = train_split()
    write_json(OUT / "manifest.json", {
        "phase": "2.6", "track": "B", "seed": SEED,
        "base_model": str(BASE), "adapter": None if ADAPTER in (None, "", "none") else str(ADAPTER),
        "train_source": str(TRAIN.relative_to(ROOT)), "candidate_source": str(CANDIDATES.relative_to(ROOT)),
        "discovery_records": len(discovery), "calibration_records": len(calibration),
        "target_layer": LAYER, "num_decoder_layers": LAYER * 2, "hidden_dim": HIDDEN,
        "target_token_cap": TOKEN_CAP, "pooling": "max over target question tokens",
        "sae": {"type": "tied-weight Top-K", "dictionary_size": DICT, "top_k": TOPK,
                "loss": "mean squared reconstruction; hard Top-K", "optimizer": "AdamW",
                "epochs": 3, "batch_size": 512, "learning_rate": 1e-3,
                "normalization": "none (raw residual-stream hidden states)"},
        "train_only_boundary": "SAE, v_j, p_anchor, p_policy use only SFT train50k; no dev labels",
        "paper_reference": "arXiv:2602.10388v4; adaptation is not exact reproduction",
        "feasible_deviation": "paper uses dictionary 2^16 (65,536); this run fixes 4,096 before development scoring",
    })
    write_jsonl(OUT / "discovery_manifest.jsonl", [{"split": "discovery", "index": i,
                 "image": r["image"], "question": r["question"]} for i, r in enumerate(discovery)])
    write_jsonl(OUT / "calibration_manifest.jsonl", [{"split": "calibration", "index": i,
                 "image": r["image"], "question": r["question"]} for i, r in enumerate(calibration)])
    write_json(OUT / "activation_cache" / "manifest.json", {"dtype": "bfloat16", "token_cap": TOKEN_CAP,
        "discovery_records": len(discovery), "calibration_records": len(calibration), "seed": SEED})


def _messages(processor, question: str) -> tuple[str, list[int]]:
    from vvqg.training.train_sft import build_messages
    full = processor.apply_chat_template(build_messages(INSTRUCTION, question), tokenize=False, add_generation_prompt=False)
    return full, processor.tokenizer.encode(question, add_special_tokens=False)


class Activations:
    def __init__(self, batch_size: int):
        from transformers import AutoModelForImageTextToText, AutoProcessor
        self.batch_size = batch_size
        self.processor = AutoProcessor.from_pretrained(str(BASE), local_files_only=True, use_fast=False)
        self.processor.tokenizer.padding_side = "right"
        model = AutoModelForImageTextToText.from_pretrained(str(BASE), dtype=torch.bfloat16,
            local_files_only=True, attn_implementation="sdpa")
        if ADAPTER in (None, "", "none"):
            self.model = model.eval().requires_grad_(False)
        else:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(model, str(ADAPTER), is_trainable=False).eval().requires_grad_(False)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    @torch.inference_mode()
    def batch(self, rows: list[dict], blank: bool = False) -> tuple[torch.Tensor, list[dict]]:
        images = []
        texts, qids = [], []
        for row in rows:
            with Image.open(ROOT / row["image"]) as im:
                rgb = im.convert("RGB")
                images.append(Image.new("RGB", rgb.size, (127, 127, 127)) if blank else rgb.copy())
            text, ids = _messages(self.processor, row["question"])
            texts.append(text); qids.append(ids)
        encoded = self.processor(text=texts, images=images, padding=True, return_tensors="pt")
        grids = encoded["image_grid_thw"]
        batch = {k: v.to(self.device) if hasattr(v, "to") else v for k, v in encoded.items()}
        hidden = self.model(**batch, use_cache=False, output_hidden_states=True).hidden_states[LAYER + 1]
        vectors, audit = [], []
        for i, ids in enumerate(qids):
            n = int(encoded["attention_mask"][i].sum())
            seq = encoded["input_ids"][i, :n].tolist()
            found = next((j for j in range(len(seq) - len(ids) + 1) if seq[j:j + len(ids)] == ids), None)
            if found is None:
                raise RuntimeError("question token alignment mismatch")
            # deterministic token-position policy: first TOKEN_CAP question tokens.
            pos = torch.arange(found, min(found + TOKEN_CAP, found + len(ids)), device=self.device)
            vectors.append(hidden[i, pos].float().cpu())
            audit.append({"question_token_ids": ids, "question_token_count": len(ids),
                          "question_start": found, "image_grid_thw": grids[i].tolist(), "blank": blank})
        for im in images: im.close()
        return torch.nn.utils.rnn.pad_sequence(vectors, batch_first=True), audit


def extract(args: argparse.Namespace) -> None:
    discovery, calibration = train_split()
    rows = discovery if args.split == "discovery" else calibration
    if args.limit is not None: rows = rows[:args.limit]
    name = f"activations_{args.split}{'_blank' if args.blank else ''}.pt"
    out = OUT / name
    act = Activations(args.batch_size)
    values, audits = [], []
    for left in range(0, len(rows), args.batch_size):
        shard_dir = OUT / "activation_cache" / "shards" / (args.split + ("_blank" if args.blank else "")); shard_dir.mkdir(parents=True, exist_ok=True)
        shard = shard_dir / f"shard_{left // args.batch_size:05d}.pt"
        if shard.exists() and not args.overwrite:
            cached = torch.load(shard, map_location="cpu")
            values.append(pad_token_axis(cached["activations"] if isinstance(cached, dict) else cached))
            if isinstance(cached, dict): audits.extend(cached["audit"])
            continue
        x, a = act.batch(rows[left:left + args.batch_size], blank=args.blank)
        values.append(pad_token_axis(x).to(torch.bfloat16)); audits.extend(a)
        torch.save({"activations": pad_token_axis(x).to(torch.bfloat16), "audit": a, "records": rows[left:left + args.batch_size], "dtype": "bfloat16", "seed": SEED, "blank": args.blank}, shard)
        print(f"extract {args.split} {min(left + args.batch_size, len(rows))}/{len(rows)}", flush=True)
    stacked = torch.cat(values, dim=0)
    shard_dir = OUT / "activation_cache" / "shards" / (args.split + ("_blank" if args.blank else "")); shard_dir.mkdir(parents=True, exist_ok=True)
    shard_rows = []
    for shard_id, chunk in enumerate(values):
        shard = shard_dir / f"shard_{shard_id:05d}.pt"
        shard_rows.append({"shard": str(shard.relative_to(ROOT)), "records": len(chunk), "dtype": "bfloat16"})
    write_json(OUT / "activation_cache" / f"{args.split}{'_blank' if args.blank else ''}_manifest.json", {"split": args.split, "blank": args.blank, "records": len(rows), "shards": shard_rows, "dtype": "bfloat16", "seed": SEED})
    torch.save({"activations": stacked, "audit": audits,
                "records": rows, "dtype": "float32", "seed": SEED,
                "blank": args.blank, "shards": shard_rows}, out)


def policy_questions() -> None:
    from vvqg.generation import GenerationConfig, VQGGenerator
    if POLICY_MANIFEST is not None:
        items = [(p, None) for p in sorted({r.get("image_path") or r["image"] for r in read(POLICY_MANIFEST)})]
    else:
        rows = read(TRAIN); by_image = {}
        for row in rows: by_image.setdefault(row["image"], row)
        items = sorted(by_image.items())
    generator = VQGGenerator(GenerationConfig(base_model=str(BASE), checkpoint=None if ADAPTER in (None, "", "none") else str(ADAPTER),
        batch_size=4, num_samples=1, temperature=0.0, max_new_tokens=32, seed=SEED, backend="pil"))
    result = []
    for left in range(0, len(items), 4):
        for image, questions in generator.generate_many([(str(i), p) for i, (p, _) in enumerate(items[left:left + 4])]):
            result.append({"image": items[left + int(image)][0], "question": questions[0]})
        print(f"policy {min(left + 4, len(items))}/{len(items)}", flush=True)
    write_jsonl(OUT / "policy_questions.jsonl", result)


def extract_policy(args: argparse.Namespace) -> None:
    rows = read(OUT / "policy_questions.jsonl")
    act = Activations(args.batch_size); values, audits = [], []
    shard_dir = OUT / "activation_cache" / "shards" / "policy"; shard_dir.mkdir(parents=True, exist_ok=True)
    for left in range(0, len(rows), args.batch_size):
        shard = shard_dir / f"shard_{left // args.batch_size:05d}.pt"
        if shard.exists():
            cached = torch.load(shard, map_location="cpu"); values.append(cached["activations"].float()); audits.extend(cached["audit"]); continue
        x, a = act.batch(rows[left:left + args.batch_size]); values.append(pad_token_axis(x).to(torch.bfloat16)); audits.extend(a)
        torch.save({"activations": pad_token_axis(x).to(torch.bfloat16), "audit": a, "records": rows[left:left + args.batch_size], "dtype": "bfloat16", "seed": SEED}, shard)
        print(f"extract policy {min(left + args.batch_size, len(rows))}/{len(rows)}", flush=True)
    stacked = torch.cat(values, dim=0)
    shard_rows = []
    for shard_id, chunk in enumerate(values):
        shard = shard_dir / f"shard_{shard_id:05d}.pt"
        shard_rows.append({"shard": str(shard.relative_to(ROOT)), "records": len(chunk), "dtype": "bfloat16"})
    write_json(OUT / "activation_cache" / "policy_manifest.json", {"split": "policy", "records": len(rows), "shards": shard_rows, "dtype": "bfloat16", "seed": SEED})
    torch.save({"activations": stacked, "audit": audits, "records": rows,
                "dtype": "float32", "seed": SEED}, OUT / "activations_policy.pt")


class TopKSAE(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty(HIDDEN, DICT))
        self.bias = torch.nn.Parameter(torch.zeros(HIDDEN))
        torch.nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = torch.relu((x - self.bias) @ self.weight)
        vals, idx = torch.topk(z, TOPK, dim=-1)
        sparse = torch.zeros_like(z).scatter(1, idx, vals)
        return sparse, sparse @ self.weight.T + self.bias


def train_sae(args: argparse.Namespace) -> None:
    path = OUT / "activations_discovery.pt"
    if path.exists():
        data = torch.load(path, map_location="cpu")
    else:
        shards = sorted((OUT / "activation_cache/shards/discovery").glob("shard_*.pt"))
        if not shards: raise RuntimeError("missing discovery activation shards")
        parts = [torch.load(p, map_location="cpu") for p in shards]
        data = {"activations": torch.cat([p["activations"].float() for p in parts]), "audit": sum((p["audit"] for p in parts), [])}
    x = data["activations"].reshape(-1, HIDDEN)
    model = TopKSAE().to("cuda" if torch.cuda.is_available() else "cpu")
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    device = next(model.parameters()).device
    losses = []
    generator = torch.Generator().manual_seed(SEED)
    for epoch in range(3):
        order = torch.randperm(len(x), generator=generator)
        total = 0.0
        for ids in order.split(512):
            xb = x[ids].to(device=device, dtype=torch.float32)
            z, recon = model(xb)
            loss = (recon - xb).pow(2).mean()
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            total += float(loss) * len(ids)
        losses.append(total / len(x)); print(f"sae epoch={epoch + 1}/3 loss={losses[-1]:.6f}", flush=True)
    torch.save({"state_dict": model.state_dict(), "losses": losses, "config": {"hidden": HIDDEN, "dictionary": DICT, "top_k": TOPK}}, OUT / "sae.pt")
    (OUT / "sae" / "checkpoint").mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "config": {"hidden": HIDDEN, "dictionary": DICT, "top_k": TOPK}}, OUT / "sae" / "checkpoint" / "model.pt")
    write_json(OUT / "sae" / "config.json", {"hidden": HIDDEN, "dictionary": DICT, "top_k": TOPK, "type": "tied-weight Top-K"})
    with torch.no_grad():
        sample = x[:min(len(x), 4096)].to(device=device, dtype=torch.float32)
        z, recon = model(sample); mse = (recon - sample).pow(2).mean().item()
        variance = sample.var().item(); explained = 1 - mse / variance if variance else None
        active = (z > 0).any(0); dead = float((~active).float().mean()); mean_active = float((z > 0).float().sum(1).mean())
    quality = {"reconstruction_loss": losses[-1], "loss_history": losses,
        "sample_reconstruction_loss": mse, "explained_variance": explained,
        "average_active_features": mean_active, "dead_feature_fraction": dead,
        "activation_sparsity": 1 - mean_active / DICT, "dictionary_size": DICT, "top_k": TOPK, "status": "trained"}
    quality["status"] = "rejected" if (not torch.isfinite(torch.tensor(losses[-1])) or dead > 0.95 or (explained is not None and explained < 0.0)) else "trained"
    write_json(OUT / "sae_quality.json", quality); write_json(OUT / "sae" / "training_metrics.json", quality)
    if quality["status"] == "rejected":
        raise RuntimeError(f"SAE quality failure: dead_feature_fraction={dead:.4f}, explained_variance={explained}")


def candidate_rows() -> list[dict]:
    rows = read(CANDIDATES)
    canon = read(CANON)
    canon_by_key = {(str(r["image_id"]), r["question"], int(r["generation_id"])): r for r in canon}
    if len(rows) != 4000:
        raise RuntimeError("candidate pool must contain exactly 4000 rows")
    for row in rows:
        key = (str(row["image_id"]), row["question"], int(row["generation_id"]))
        if key not in canon_by_key:
            raise RuntimeError("candidate pool differs from Phase-2.5 canonical keys")
        row["candidate_id"] = canon_by_key[key]["candidate_id"]
    if len({r["candidate_id"] for r in rows}) != 4000:
        raise RuntimeError("candidate pool must contain 4000 unique candidate IDs")
    return rows


def _load_sae() -> TopKSAE:
    model = TopKSAE(); model.load_state_dict(torch.load(OUT / "sae.pt", map_location="cpu")["state_dict"])
    return model.eval()


def calibrate(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sae = _load_sae().to(device).eval()
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True

    def freq(path: Path, label: str) -> torch.Tensor:
        data = torch.load(path, map_location="cpu")
        x = data["activations"]
        audits = data["audit"]
        total = len(audits)
        token_cap = int(x.shape[1])
        hidden = int(x.shape[2])
        block = 4096
        counts = torch.zeros(DICT, device=device, dtype=torch.float32)
        ranges = range(0, total, block)
        with torch.inference_mode():
            for start in tqdm(
                ranges,
                total=(total + block - 1) // block,
                desc=f"calibrate {label}",
                unit="batch",
                dynamic_ncols=True,
            ):
                ends = min(start + block, total)
                n = ends - start
                flat = x[start:ends].reshape(-1, hidden).to(
                    device=device, dtype=torch.float32, non_blocking=True
                )
                z, _ = sae(flat)
                z = z.view(n, token_cap, DICT)
                lengths = torch.tensor(
                    [
                        min(max(int(a.get("question_token_count", token_cap)), 0), token_cap)
                        for a in audits[start:ends]
                    ],
                    device=device,
                    dtype=torch.long,
                )
                token_mask = torch.arange(token_cap, device=device)[None, :] < lengths[:, None]
                pooled = z.masked_fill(~token_mask[:, :, None], 0).amax(dim=1)
                counts.add_((pooled > 0).sum(dim=0))
                del flat, z, pooled, token_mask, lengths
        return counts.cpu() / total

    p_real = freq(OUT / "activations_calibration.pt", "real")
    p_blank = freq(OUT / "activations_calibration_blank.pt", "blank")
    v = (p_real - p_blank).clamp_min(0)
    p_anchor = freq(OUT / "activations_discovery.pt", "anchor")
    policy_path = OUT / "activations_policy.pt"
    if not policy_path.exists():
        raise RuntimeError("missing train-only greedy policy activations; run policy then extract-policy")
    p_policy = freq(policy_path, "policy")
    eps = 1e-6
    u = torch.log((p_anchor + eps) / (p_policy + eps)).clamp_min(0)
    src = "raw base model" if ADAPTER is None else "SFT-adapted model"
    torch.save(
        {"p_real": p_real, "p_blank": p_blank, "v": v, "p_anchor": p_anchor,
         "p_policy": p_policy, "u": u, "eps": eps, "source": f"{src}, SFT train50k data"},
        OUT / "weights.pt",
    )
    write_json(
        OUT / "weights_summary.json",
        {"features": DICT, "visual_sensitive": int((v > 0).sum()),
         "undercovered": int((u > 0).sum()), "eps": eps, "source": f"{src}, SFT train50k data"},
    )
    write_json(
        OUT / "visual_calibration" / "feature_weights.json",
        {"v": v.tolist(), "p_real": p_real.tolist(), "p_blank": p_blank.tolist(),
         "source": f"{src}, SFT train calibration"},
    )
    write_json(
        OUT / "anchor" / "human_feature_frequency.json",
        {"p_anchor": p_anchor.tolist(), "source": f"{src}, SFT train discovery"},
    )
    write_json(
        OUT / "anchor" / "policy_feature_frequency.json",
        {"p_policy": p_policy.tolist(), "source": f"{src}, train greedy policy"},
    )
    write_json(
        OUT / "anchor" / "undercoverage_weights.json",
        {"u": u.tolist(), "eps": eps, "source": f"{src}, SFT train"},
    )


def score(args: argparse.Namespace) -> None:
    rows = candidate_rows()
    weights = torch.load(OUT / "weights.pt", map_location="cpu")
    sae = _load_sae()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sae = sae.to(device).eval()
    out = OUT / "scores4000.jsonl"
    existing = {r["candidate_id"]: r for r in read(out)} if out.exists() else {}
    pending = [r for r in rows if r["candidate_id"] not in existing]
    act = Activations(args.batch_size)
    v_weights = weights["v"].to(device)
    u_weights = weights["u"].to(device)
    started = time.perf_counter()
    with out.open("a") as fh, torch.inference_mode():
        for left in tqdm(
            range(0, len(pending), args.batch_size),
            total=(len(pending) + args.batch_size - 1) // args.batch_size,
            desc="score",
            unit="batch",
            dynamic_ncols=True,
        ):
            chunk = pending[left:left + args.batch_size]
            x, audits = act.batch(chunk)
            n, token_cap, hidden = x.shape
            flat = x.reshape(-1, hidden).to(device=device, dtype=torch.float32, non_blocking=True)
            z, _ = sae(flat)
            z = z.view(n, token_cap, DICT)
            lengths = torch.tensor(
                [
                    min(max(int(a.get("question_token_count", token_cap)), 0), token_cap)
                    for a in audits
                ],
                device=device,
                dtype=torch.long,
            )
            mask = torch.arange(token_cap, device=device)[None, :] < lengths[:, None]
            pooled = z.masked_fill(~mask[:, :, None], 0).amax(dim=1)
            active = pooled > 0
            vfc = (active * v_weights[None, :]).sum(dim=1)
            uvfc = (active * (v_weights * u_weights)[None, :]).sum(dim=1)
            for row, audit, active_row, vfc_row, uvfc_row in zip(
                chunk, audits, active.cpu(), vfc.cpu(), uvfc.cpu()
            ):
                result = {
                    "candidate_id": row["candidate_id"],
                    "image_id": row["image_id"],
                    "image": row["image"],
                    "question": row["question"],
                    "question_token_count": audit["question_token_count"],
                    "active_features": torch.where(active_row)[0].tolist(),
                    "active_visual_weighted_feature_count": int((active_row & (weights["v"] > 0)).sum()),
                    "VFC": float(vfc_row),
                    "UVFC": float(uvfc_row),
                    "layer": LAYER,
                    "hidden_dim": HIDDEN,
                    "sae_dictionary": DICT,
                    "sae_top_k": TOPK,
                }
                fh.write(json.dumps(result, ensure_ascii=False) + "\n")
            fh.flush()
            del flat, z, pooled, active, mask, lengths
    final = {r["candidate_id"]: r for r in read(out)}
    if set(final) != {r["candidate_id"] for r in rows}:
        raise RuntimeError("VFC coverage mismatch")
    runtime = {
        "records": len(final), "signals": ["VFC", "UVFC"], "seed": SEED,
        "wall_seconds": time.perf_counter() - started, "batch_size": args.batch_size,
        "model": str(BASE), "adapter": str(ADAPTER), "layer": LAYER,
        "hidden_dim": HIDDEN,
    }
    write_json(OUT / "score_summary.json", runtime)
    write_json(OUT / "runtime.json", runtime)


def main() -> None:
    global ADAPTER, OUT, BASE, TRAIN, POLICY_MANIFEST
    p = argparse.ArgumentParser(); p.add_argument("stage", choices=["manifest", "extract", "train", "policy", "extract-policy", "calibrate", "score"])
    p.add_argument("--split", choices=["discovery", "calibration"], default="discovery"); p.add_argument("--blank", action="store_true")
    p.add_argument("--batch-size", type=int, default=4); p.add_argument("--limit", type=int, default=None); p.add_argument("--overwrite", action="store_true")
    p.add_argument("--adapter", default=str(ADAPTER), help="SFT adapter path, or 'none' to extract from the raw base model")
    p.add_argument("--out", type=Path, default=OUT, help="VFC output directory (default: frozen outputs/phase26/vfc)")
    p.add_argument("--base", default=str(BASE), help="base model path (default: frozen Qwen3-VL-8B-Instruct)")
    p.add_argument("--train", type=Path, default=TRAIN, help="SFT train manifest (default: frozen VQAv2 train.jsonl)")
    p.add_argument("--policy-manifest", type=Path, default=None, help="optional policy image manifest (default: unique images of --train)")
    args = p.parse_args()
    ADAPTER = None if str(args.adapter).lower() in ("none", "") else args.adapter
    OUT = args.out if args.out.is_absolute() else ROOT / args.out
    BASE = str(args.base)
    TRAIN = args.train if args.train.is_absolute() else ROOT / args.train
    POLICY_MANIFEST = None if args.policy_manifest is None else (args.policy_manifest if args.policy_manifest.is_absolute() else ROOT / args.policy_manifest)
    OUT.mkdir(parents=True, exist_ok=True); configure_model()
    if args.stage == "manifest": manifest()
    elif args.stage == "extract": extract(args)
    elif args.stage == "train": train_sae(args)
    elif args.stage == "policy": policy_questions()
    elif args.stage == "extract-policy": extract_policy(args)
    elif args.stage == "calibrate": calibrate(args)
    else: score(args)


if __name__ == "__main__": main()
