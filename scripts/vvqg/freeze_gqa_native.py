#!/usr/bin/env python3
"""Phase G-NATIVE-0: freeze the GQA-native VQG experiment data contract.

No model is run. Builds GQA-SFT-50K, the shared method-training image manifest,
DEV1000, SANITY50, and the read-only TEST1000 reference, plus split integrity.

Run with the vllm env (numpy, scipy, PIL):
    .venv-vllm-cu128-v011/bin/python scripts/vvqg/freeze_gqa_native.py
"""

from __future__ import annotations

import collections
import json
import os
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "src"))
from freeze_data_contract import _hash_many, sha256_file  # noqa: E402
from vvqg.baselines.common import SEED, write_jsonl  # noqa: E402

GQA_Q = ROOT / "data/raw/gqa/questions1.2"
GQA_IMG = ROOT / "data/raw/gqa/images_pool"
OUT = ROOT / "outputs/gqa_native/manifests"
G0_IMAGES = ROOT / "outputs/data_contract/gqa_generalization1000_images.jsonl"
G0_REFS = ROOT / "outputs/data_contract/gqa_generalization1000_refs.json"

SFT_N = 50_000
METHOD_N = 10_000
DEV_N = 1_000
TEST_N = 1_000
SANITY_N = 50


def image_path(image_id: str) -> str:
    return f"data/raw/gqa/images_pool/{image_id}.jpg"


def sort_key(image_id: str):
    return int(image_id[1:]) if image_id.startswith("n") else int(image_id)


def load_split(name: str) -> list[tuple[str, str, str]]:
    data = json.loads((GQA_Q / name).read_text())
    rows = [(qid, rec["imageId"], rec["question"])
            for qid, rec in data.items() if rec.get("isBalanced", True)]
    rows.sort()
    return rows


def refs_by_image(rows: list[tuple[str, str, str]]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = collections.defaultdict(list)
    for _, image_id, question in rows:
        grouped[image_id].append(question)
    return {image_id: sorted(set(questions)) for image_id, questions in grouped.items()}


def sample(items: list[str], n: int) -> list[str]:
    return sorted(random.Random(SEED).sample(sorted(items, key=sort_key), n))


def main() -> int:
    started = time.time()
    (OUT).mkdir(parents=True, exist_ok=True)

    train_rows = load_split("train_balanced_questions.json")
    val_rows = load_split("val_balanced_questions.json")
    test_rows = load_split("test_balanced_questions.json")

    # ---- integrity of the full official split -----------------------------
    for tag, rows in (("train", train_rows), ("val", val_rows), ("test", test_rows)):
        assert rows, f"{tag} split empty"
        assert all(image_id and question.strip() for _, image_id, question in rows), f"{tag} invalid pair"
    train_images = {image_id for _, image_id, _ in train_rows}
    val_images = {image_id for _, image_id, _ in val_rows}
    test_images = {image_id for _, image_id, _ in test_rows}
    missing = [i for i in train_images | val_images | test_images if not (GQA_IMG / f"{i}.jpg").exists()]
    assert not missing, f"missing image files: {missing[:5]}"
    assert not (train_images & val_images), "train/val image leakage"
    assert not (train_images & test_images), "train/test image leakage"
    assert not (val_images & test_images), "val/test image leakage"

    # ---- GQA-SFT-50K -------------------------------------------------------
    sft_pairs = random.Random(SEED).sample(train_rows, SFT_N)
    sft_records = [{"image_id": image_id, "image_path": image_path(image_id), "question": question}
                   for _, image_id, question in sft_pairs]
    exposure = sorted({rec["image_id"] for rec in sft_records}, key=sort_key)
    exposure_records = [{"image_id": i, "image_path": image_path(i),
                         "source": "GQA", "split": "train_balanced"} for i in exposure]

    # ---- shared method-training manifest -----------------------------------
    method_n = METHOD_N if len(exposure) >= METHOD_N else len(exposure)
    method_images = sample(exposure, method_n)
    method_records = [{"image_id": i, "image_path": image_path(i),
                       "source": "GQA", "split": "train_balanced"} for i in method_images]

    # ---- DEV1000 -----------------------------------------------------------
    val_refs = refs_by_image(val_rows)
    dev_images = sample(list(val_refs), DEV_N)
    sanity_pool = [i for i in val_refs if i not in set(dev_images)]
    sanity_images = sample(sanity_pool, SANITY_N)
    assert not (set(dev_images) & set(sanity_images))

    # ---- TEST1000 (read-only copy of the frozen G0 set) --------------------
    g0_images = [json.loads(line) for line in G0_IMAGES.read_text().splitlines() if line.strip()]
    g0_refs = json.loads(G0_REFS.read_text())
    g0_ids = [r["image_id"] for r in g0_images]
    assert len(g0_ids) == TEST_N and len(set(g0_ids)) == TEST_N
    assert list(g0_refs) == g0_ids
    assert set(g0_ids) <= test_images, "G0 test500 not from test_balanced"
    test_expected_refs = refs_by_image([r for r in test_rows if r[1] in set(g0_ids)])
    assert all(g0_refs[i]["references"] == test_expected_refs[i] for i in g0_ids), "G0 refs mismatch"
    reconstructed = sorted(random.Random(SEED).sample(sorted(g0_ids, key=sort_key), TEST_N))
    assert reconstructed == g0_ids, "G0 test1000 is not the seed-42 frozen sample"

    # ---- decoded-identity check over selected images -----------------------
    selected = {}
    for i in exposure:
        selected[i] = image_path(i)
    for i in dev_images + sanity_images + g0_ids:
        selected[i] = image_path(i)
    hashed = _hash_many(sorted(selected.values()))
    by_file: dict[str, list[str]] = collections.defaultdict(list)
    by_rgb: dict[str, list[str]] = collections.defaultdict(list)
    for image_id, rel in selected.items():
        by_file[hashed[rel]["file_sha256"]].append(image_id)
        by_rgb[hashed[rel]["rgb_sha256"]].append(image_id)
    collisions = [{"kind": kind, "images": sorted(ids), "hash": h}
                  for kind, table in (("sha256_file", by_file), ("normalized_rgb", by_rgb))
                  for h, ids in table.items() if len(ids) > 1]
    val_ids = set(dev_images) | set(sanity_images)
    test_ids = set(g0_ids)
    role = lambda i: "test" if i in test_ids else ("val" if i in val_ids else "train")
    for c in collisions:
        c["roles"] = sorted({role(i) for i in c["images"]})
    cross_split = [c for c in collisions if len(c["roles"]) > 1]
    (OUT / "split_image_hash_collisions.json").write_text(json.dumps(collisions, indent=2))

    # ---- write manifests ---------------------------------------------------
    write_jsonl(OUT / "sft_train_50000.jsonl", sft_records)
    write_jsonl(OUT / "train_exposure_images.jsonl", exposure_records)
    write_jsonl(OUT / "method_train_10000.jsonl", method_records)
    write_jsonl(OUT / "dev1000_images.jsonl",
                [{"image_id": i, "image_path": image_path(i), "source": "GQA", "split": "val_balanced"}
                 for i in dev_images])
    write_jsonl(OUT / "sanity50_images.jsonl",
                [{"image_id": i, "image_path": image_path(i), "source": "GQA", "split": "val_balanced"}
                 for i in sanity_images])
    (OUT / "dev1000_refs.json").write_text(json.dumps(
        {i: {"references": val_refs[i]} for i in dev_images}, ensure_ascii=False, indent=1))
    (OUT / "sanity50_refs.json").write_text(json.dumps(
        {i: {"references": val_refs[i]} for i in sanity_images}, ensure_ascii=False, indent=1))
    for src, dst in ((G0_IMAGES, OUT / "test1000_images.jsonl"), (G0_REFS, OUT / "test1000_refs.json")):
        (dst).write_bytes(src.read_bytes())

    # ---- manifest ----------------------------------------------------------
    def describe(path: Path, id_key: str) -> dict:
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        return {"path": str(path.relative_to(ROOT)), "records": len(rows),
                "unique_images": len({r[id_key] for r in rows}), "sha256": sha256_file(path)}

    manifest = {
        "schema": "gqa-native-data-contract-v1",
        "phase": "G-NATIVE-0",
        "seed": SEED,
        "creation_script": "scripts/vvqg/freeze_gqa_native.py",
        "git_commit": os.popen("git rev-parse HEAD").read().strip(),
        "question_source": {
            "train_balanced_questions.json": sha256_file(GQA_Q / "train_balanced_questions.json"),
            "val_balanced_questions.json": sha256_file(GQA_Q / "val_balanced_questions.json"),
            "test_balanced_questions.json": sha256_file(GQA_Q / "test_balanced_questions.json"),
        },
        "files": {
            "sft_train_50000.jsonl": describe(OUT / "sft_train_50000.jsonl", "image_id"),
            "train_exposure_images.jsonl": describe(OUT / "train_exposure_images.jsonl", "image_id"),
            "method_train_10000.jsonl": describe(OUT / "method_train_10000.jsonl", "image_id"),
            "dev1000_images.jsonl": describe(OUT / "dev1000_images.jsonl", "image_id"),
            "sanity50_images.jsonl": describe(OUT / "sanity50_images.jsonl", "image_id"),
            "dev1000_refs.json": {"path": "outputs/gqa_native/manifests/dev1000_refs.json",
                                  "records": len(dev_images), "sha256": sha256_file(OUT / "dev1000_refs.json")},
            "sanity50_refs.json": {"path": "outputs/gqa_native/manifests/sanity50_refs.json",
                                   "records": len(sanity_images), "sha256": sha256_file(OUT / "sanity50_refs.json")},
            "test1000_images.jsonl": {"path": "outputs/gqa_native/manifests/test1000_images.jsonl",
                                      "records": len(g0_ids), "unique_images": len(g0_ids),
                                      "sha256": sha256_file(OUT / "test1000_images.jsonl"),
                                      "source": str(G0_IMAGES.relative_to(ROOT)),
                                      "source_sha256": sha256_file(G0_IMAGES)},
            "test1000_refs.json": {"path": "outputs/gqa_native/manifests/test1000_refs.json",
                                   "records": len(g0_ids), "sha256": sha256_file(OUT / "test1000_refs.json"),
                                   "source": str(G0_REFS.relative_to(ROOT)), "source_sha256": sha256_file(G0_REFS)},
        },
        "counts": {
            "sft_records": len(sft_records),
            "sft_unique_images": len(exposure),
            "method_train_images": len(method_images),
            "dev_images": len(dev_images),
            "sanity_images": len(sanity_images),
            "test_images": len(g0_ids),
        },
        "intersections": {
            "train_val": len(train_images & val_images),
            "train_test": len(train_images & test_images),
            "val_test": len(val_images & test_images),
            "exposure_dev": len(set(exposure) & set(dev_images)),
            "exposure_test": len(set(exposure) & set(g0_ids)),
            "dev_sanity": len(set(dev_images) & set(sanity_images)),
            "dev_test": len(set(dev_images) & set(g0_ids)),
        },
        "decoded_identity_collisions": collisions,
        "cross_split_collisions": cross_split,
        "status": "FROZEN",
    }
    (OUT / "data_contract_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))

    assert manifest["counts"]["sft_records"] == SFT_N
    assert manifest["counts"]["test_images"] == TEST_N and manifest["counts"]["dev_images"] == DEV_N
    assert not any(manifest["intersections"].values()), manifest["intersections"]
    assert not cross_split, f"cross-split decoded-identical images: {cross_split[:3]}"

    print(f"GQA NATIVE DATA CONTRACT FROZEN in {time.time() - started:.1f}s")
    print(json.dumps(manifest["counts"], ensure_ascii=False))
    print(json.dumps(manifest["intersections"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
