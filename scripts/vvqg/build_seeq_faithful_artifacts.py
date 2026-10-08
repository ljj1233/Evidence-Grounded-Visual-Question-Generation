#!/usr/bin/env python3
"""Emit the frozen SeeQ-faithful pre-flight artifacts: split audit + prompts."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from vvqg.baselines import seeq_faithful as SF  # noqa: E402

OUT = ROOT / "outputs" / "same_backbone_baselines" / "seeq_faithful"
MANIFEST_DIR = ROOT / "outputs" / "same_backbone_baselines" / "manifests"


def ids(path: Path) -> set[str]:
    return {str(json.loads(line)["image_id"]) for line in path.read_text().splitlines() if line.strip()}


def digest(values: set[str]) -> str:
    return hashlib.sha256("\n".join(sorted(values)).encode()).hexdigest()


def split_audit() -> dict:
    splits = {
        "train_unlabeled": ids(MANIFEST_DIR / "train_unlabeled_images.jsonl"),
        "development1000": ids(MANIFEST_DIR / "development1000_images.jsonl"),
        "reward_calibration": ids(ROOT / "outputs/evaluation/sft_greedy/val2014_sample1000.jsonl"),
        "final_holdout": ids(ROOT / "src/vvqg/data/evaluation/final_holdout_manifest.jsonl"),
    }
    names = list(splits)
    overlaps = {
        f"{a}__{b}": len(splits[a] & splits[b])
        for i, a in enumerate(names) for b in names[i + 1:]
    }
    return {
        "schema": "seeq-faithful-split-audit-v1",
        "seed": 42,
        "counts": {k: len(v) for k, v in splits.items()},
        "sha256": {k: digest(v) for k, v in splits.items()},
        "overlaps": overlaps,
        "train_vs_dev_overlap": overlaps["train_unlabeled__development1000"],
        "train_vs_final_holdout_overlap": overlaps["train_unlabeled__final_holdout"],
        "train_vs_reward_calibration_overlap": overlaps["train_unlabeled__reward_calibration"],
        "dev_equals_reward_calibration": splits["development1000"] == splits["reward_calibration"],
        "all_train_relations_disjoint": all(
            overlaps[f"train_unlabeled__{other}"] == 0
            for other in ("development1000", "reward_calibration", "final_holdout")
        ),
    }


def proposal_prompts() -> dict:
    records = []
    for intent, condition in SF.INTENTS.items():
        records.append({
            "intent": intent,
            "prompt": SF.proposal_prompt(intent),
            "intent_condition": condition,
            "source": "LOCAL_CONTROLLED_ADAPTATION",
            "base_prompt_source": "PAPER",
            "source_location": "PAPER Appendix B Fig.2 == SeeQ utils/gen/evol_utils.py:genqa_prompt_v1; intent wording local (paper names intents without templates, PAPER 3.1)",
        })
    return {
        "schema": "seeq-faithful-proposal-prompts-v1",
        "intents": list(SF.INTENTS),
        "proposals_per_image": 10,
        "proposals_per_intent": 2,
        "prompts": records,
    }


def rewrite_prompt() -> dict:
    records = []
    for intent, prompt in SF.prompts.INTENT_REWRITE.items():
        records.append({
            "intent": intent,
            "prompt": prompt,
            "source": "OFFICIAL_CODE",
            "source_location": "SeeQ utils/gen/evol_utils.py question_condition/rewrite_format + strategy prompt",
        })
    return {
        "schema": "seeq-faithful-rewrite-prompts-v1",
        "model": "frozen M0",
        "answer_model": "frozen M0",
        "prompts": records,
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    audit = split_audit()
    assert audit["train_vs_dev_overlap"] == 0, audit
    assert audit["train_vs_final_holdout_overlap"] == 0, audit
    (OUT / "split_audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    (OUT / "proposal_prompts.json").write_text(json.dumps(proposal_prompts(), indent=2, sort_keys=True) + "\n")
    (OUT / "rewrite_prompt.json").write_text(json.dumps(rewrite_prompt(), indent=2, sort_keys=True) + "\n")
    print(json.dumps({"counts": audit["counts"], "overlaps": audit["overlaps"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
