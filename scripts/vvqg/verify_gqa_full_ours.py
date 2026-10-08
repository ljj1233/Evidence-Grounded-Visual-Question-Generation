#!/usr/bin/env python3
"""Read-only integrity check for the GQA-native Full Ours chain.

Validates that each present artifact is *complete and provenance-matched*, not
merely that a file exists.  Exits non-zero on any mismatch so a guarded launcher
can block downstream stages.  Loads no model.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from vvqg.paths import local_model_path

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "outputs/gqa_native/full_ours"
DATA = BASE / "data"
UVFC = BASE / "uvfc"
SFT = BASE / "sft/checkpoint-final"
D4 = BASE / "direct4"
D1 = BASE / "distill_d1/checkpoint-final"
G2 = BASE / "g2"
EXPECTED_BASE = local_model_path('Qwen3-VL-8B-Instruct')
METHOD_IMAGES = 10000


def read(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            out.append(json.loads(line))
    return out


def check_adapter(path: Path, label: str, errs: list[str], required: bool) -> None:
    cfg, weights = path / "adapter_config.json", path / "adapter_model.safetensors"
    if not cfg.is_file() or not weights.is_file():
        if required:
            errs.append(f"{label}: missing adapter_config.json or adapter_model.safetensors")
        return
    payload = json.loads(cfg.read_text())
    if payload.get("base_model_name_or_path") != EXPECTED_BASE:
        errs.append(f"{label}: base_model {payload.get('base_model_name_or_path')!r} != {EXPECTED_BASE!r}")
    if weights.stat().st_size == 0:
        errs.append(f"{label}: adapter weights are empty")
    try:
        from peft import PeftConfig
        PeftConfig.from_pretrained(str(path))
    except Exception as error:  # real external boundary: a loadability check
        errs.append(f"{label}: PeftConfig.from_pretrained failed ({type(error).__name__})")


def check_uvfc(errs: list[str]) -> None:
    import torch

    if not (UVFC / "sae.pt").is_file() or not (UVFC / "weights.pt").is_file():
        errs.append("uvfc: missing sae.pt or weights.pt")
        return
    w = torch.load(UVFC / "weights.pt", map_location="cpu")
    for key in ("v", "u", "p_anchor", "p_policy"):
        if key not in w:
            errs.append(f"uvfc: weights.pt missing {key}")
        elif not torch.isfinite(w[key]).all():
            errs.append(f"uvfc: weights.pt {key} has non-finite values")
    if int((w.get("v", torch.zeros(1)) > 0).sum()) == 0 or int((w.get("u", torch.zeros(1)) > 0).sum()) == 0:
        errs.append("uvfc: v or u is identically zero (no signal)")
    sae = torch.load(UVFC / "sae.pt", map_location="cpu")
    if "state_dict" not in sae or "weight" not in sae["state_dict"]:
        errs.append("uvfc: sae.pt missing weight")


def check_direct4(errs: list[str]) -> None:
    rows = read(D4 / "candidates.jsonl")
    if not rows:
        errs.append("direct4: candidates.jsonl missing/empty")
        return
    if any(not str(r.get("question", "")).strip() or "image_id" not in r for r in rows):
        errs.append("direct4: empty question or missing image_id")
    covered = len({str(r["image_id"]) for r in rows})
    if covered != METHOD_IMAGES:
        errs.append(f"direct4: covered {covered}/{METHOD_IMAGES} method images")


def check_score(errs: list[str], require_complete: bool) -> None:
    cand = read(D4 / "candidates.jsonl")
    if not cand:
        errs.append("score: no candidate pool")
        return
    scored = D4 / "candidates_scored.jsonl"
    if not scored.is_file():
        if require_complete:
            errs.append("score: candidates_scored.jsonl missing")
        return
    lines = scored.read_text(encoding="utf-8").splitlines()
    keys = set()
    for line in lines:
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            errs.append("score: corrupted line (partial write)")
            return
        keys.add((str(r["image_id"]), int(r["generation_id"])))
    expected = {(str(r["image_id"]), int(r["generation_id"])) for r in cand}
    if require_complete and keys != expected:
        errs.append(f"score: coverage {len(keys)}/{len(expected)} (incomplete)")
    if keys - expected:
        errs.append("score: scored rows not in candidate pool")


def check_manifests(errs: list[str]) -> None:
    meth = read(DATA / "method_train_10000.jsonl")
    if len(meth) != METHOD_IMAGES:
        errs.append(f"data: method manifest has {len(meth)} rows, expected {METHOD_IMAGES}")
    if any("image" not in r for r in meth):
        errs.append("data: method manifest rows lack the 'image' key")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--phase", required=True,
                   choices=["pre-downstream", "post-score", "post-d1", "post-g2"])
    args = p.parse_args()
    errs: list[str] = []
    check_manifests(errs)

    if args.phase == "pre-downstream":
        check_adapter(SFT, "sft", errs, required=False)
        if (UVFC / "weights.pt").is_file():
            check_uvfc(errs)
        if (D4 / "candidates.jsonl").is_file():
            check_direct4(errs)
        if args.phase == "pre-downstream":
            check_score(errs, require_complete=False)
    elif args.phase == "post-score":
        check_adapter(SFT, "sft", errs, required=True)
        check_uvfc(errs)
        check_direct4(errs)
        check_score(errs, require_complete=True)
    elif args.phase == "post-d1":
        check_adapter(D1, "d1", errs, required=True)
    elif args.phase == "post-g2":
        summary_path = G2 / "training_summary.json"
        if not summary_path.is_file():
            errs.append("g2: training_summary.json missing")
        else:
            summary = json.loads(summary_path.read_text())
            for key in ("steps", "stop_reason", "checkpoints"):
                if key not in summary:
                    errs.append(f"g2: training_summary.json lacks {key}")

    if errs:
        print("VERIFY_FAIL")
        for e in errs:
            print(" -", e)
        return 1
    print("VERIFY_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
