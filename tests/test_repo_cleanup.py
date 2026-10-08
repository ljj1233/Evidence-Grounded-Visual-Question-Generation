"""Phase 4A.5 repository-hygiene regression tests (no model load)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def test_rft_stage_requires_explicit_sft_checkpoint():
    from vvqg.training import train_sft

    with pytest.raises(SystemExit):
        train_sft.require_rft_sft_checkpoint({"stage": "rft"}, None)
    train_sft.require_rft_sft_checkpoint({"stage": "rft"}, Path("/some/final_sft"))
    train_sft.require_rft_sft_checkpoint({"stage": "sft"}, None)


def test_reporting_helpers_are_deterministic():
    from vvqg.evaluation.common.reporting import bootstrap_pair, dim_vector, norm_text, read_jsonl, system_row, support_vector

    assert norm_text("  A  B ") == "a b"
    assert norm_text("ＡＢ") == "ab"  # NFKC folds full-width
    assert read_jsonl(ROOT / "does/not/exist.jsonl") == []


def test_reporting_system_row_and_bootstrap(tmp_path):
    from vvqg.evaluation.common import reporting

    greedy = tmp_path / "greedy.jsonl"
    seeq = tmp_path / "seeq.jsonl"
    support = tmp_path / "support.jsonl"
    traditional = tmp_path / "trad.json"
    diversity = tmp_path / "div.json"
    greedy.write_text("\n".join(json.dumps({"image_id": str(i), "question": f"Q{i}?"}) for i in range(3)) + "\n")
    dims = {d: 5 for d in reporting.DIMS.values()}
    seeq.write_text("\n".join(json.dumps({"image_id": str(i), "status": "ok", "scores": dims}) for i in range(3)) + "\n")
    support.write_text("\n".join(json.dumps({"image_id": str(i), "status": "ok", "overall_visual_support": "supported"}) for i in range(3)) + "\n")
    traditional.write_text(json.dumps({"corpus_bleu4": 0.1, "corpus_cider": 0.2, "corpus_qbleu4_variant": 0.3, "normalized_excess_duplicate_rate": 0.4}))
    diversity.write_text(json.dumps({"systems": {"sft": {"mean": 0.5}}}))

    paths = {"greedy": greedy, "seeq": seeq, "support": support, "traditional": traditional, "diversity": diversity}
    row = reporting.system_row(paths)
    assert row["BLEU"] == 0.1 and row["CIDEr"] == 0.2 and row["QBLEU"] == 0.3
    assert row["Search"] == 1.0 and row["Supported"] == 1.0 and row["SemDiv"] == 0.5
    pair = reporting.bootstrap_pair(paths, paths)
    assert pair["Search"]["mean_diff"] == 0.0


def test_needs_request_retry_policy():
    from vvqg.evaluation.common.cache import needs_request

    cached = {"ok": {"status": "ok"}, "err": {"status": "error"}}
    assert needs_request("missing", cached) is True
    assert needs_request("ok", cached) is False
    assert needs_request("ok", cached, retry_failures=True) is False
    assert needs_request("err", cached) is False
    assert needs_request("err", cached, retry_failures=True) is True
