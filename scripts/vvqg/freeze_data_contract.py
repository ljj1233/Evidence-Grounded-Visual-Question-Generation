#!/usr/bin/env python3
"""Phase G0: freeze the cross-dataset data contract (VQAv2 train exposure -> GQA zero-shot).

No model is run. Builds the VQAv2-derived training-exposure manifest and the
image-level GQA sanity/generalization manifests with image-level decontamination.

Run with the vllm env (has numpy, scipy, PIL):
    .venv-vllm-cu128-v011/bin/python scripts/vvqg/freeze_data_contract.py
"""

from __future__ import annotations

import collections
import hashlib
import json
import os
import random
import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.fftpack import dct

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from vvqg.baselines.common import SEED, read_jsonl, stable_hash, write_jsonl  # noqa: E402

OUT = ROOT / "outputs/data_contract"
GQA_QUESTIONS = ROOT / "data/raw/gqa/questions1.2"
GQA_IMAGES = ROOT / "data/raw/gqa/images_pool"
GQA_ARCHIVES = ROOT / "data/raw/archives/gqa"

SFT_TRAIN = ROOT / "src/vvqg/data/sft/train_manifest.jsonl"
DEV_MANIFEST = ROOT / "outputs/same_backbone_baselines/manifests/development1000_images.jsonl"
TRAIN_10000 = ROOT / "outputs/phase4b_online_uvfc/data/train_10000.jsonl"
FINAL_HOLDOUT = ROOT / "src/vvqg/data/evaluation/final_holdout_manifest.jsonl"

KNOWN_SHA256 = {
    "outputs/phase4b_online_uvfc/data/train_10000.jsonl":
        "c884ed360196e494076ab0b73a45356de87c44d5fd8b2d5778102ea25a67adc2",
    "outputs/same_backbone_baselines/manifests/development1000_images.jsonl":
        "06782bedf1754fe59ff7609a06e2c79ce91403e4ecfcf9f2f295d6b104d27d47",
}

NEAR_DUP_MAX_DISTANCE = 4
SECOND_LEVEL_CORR = 0.90
GEN_N = 1000
SANITY_N = 50


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _hash_image(rel: str) -> tuple[str, dict]:
    path = ROOT / rel
    file_sha = sha256_file(path)
    with Image.open(path) as raw:
        im = raw.convert("RGB")
    w, h = im.size
    rgb_sha = hashlib.sha256(f"{w}x{h}".encode() + im.tobytes()).hexdigest()
    gray = np.asarray(im.convert("L").resize((32, 32), Image.Resampling.LANCZOS), dtype=float)
    low = dct(dct(gray, axis=0), axis=1)[:8, :8]
    bits = (low > np.median(low)).flatten()
    phash = 0
    for bit in bits:
        phash = (phash << 1) | int(bit)
    return rel, {"file_sha256": file_sha, "rgb_sha256": rgb_sha,
                 "width": w, "height": h, "phash": phash}


def _hash_many(rel_paths: list[str]) -> dict[str, dict]:
    index: dict[str, dict] = {}
    with ProcessPoolExecutor(max_workers=min(48, os.cpu_count() or 8)) as pool:
        for rel, info in pool.map(_hash_image, rel_paths, chunksize=64):
            index[rel] = info
    return index


def _second_level_check(pair: tuple[str, str]) -> dict:
    im_a = Image.open(ROOT / pair[0]).convert("L")
    im_b = Image.open(ROOT / pair[1]).convert("L")
    a = np.asarray(im_a.resize((256, 256), Image.Resampling.BILINEAR), float)
    b = np.asarray(im_b.resize((256, 256), Image.Resampling.BILINEAR), float)
    ar_a = im_a.width / im_a.height
    ar_b = im_b.width / im_b.height
    return {"corr": round(float(np.corrcoef(a.ravel(), b.ravel())[0, 1]), 4),
            "candidate_wh": [im_a.width, im_a.height], "project_wh": [im_b.width, im_b.height],
            "aspect_ratio_rel_diff": round(abs(ar_a - ar_b) / max(ar_a, ar_b), 4)}


def image_basename(row: dict) -> str:
    return os.path.basename(str(row.get("image") or row.get("image_id") or row.get("image_path")))


def build_train_exposure() -> list[dict]:
    train = read_jsonl(SFT_TRAIN)
    stages = {
        "uvfc": [ROOT / "outputs/phase4b_online_uvfc/data/train_10000.jsonl",
                 ROOT / "outputs/phase4b_online_uvfc/data/train_2000.jsonl"],
        "teacher_generation": [ROOT / "outputs/phase5b_teacher_explore_ground/train_10000.jsonl"],
        "distillation": [ROOT / "outputs/phase5c_qse_teacher/targets_d1_single.jsonl",
                         ROOT / "outputs/phase5c_qse_teacher/targets_d2_multipos.jsonl",
                         ROOT / "outputs/phase5b_teacher_explore_ground/student_train_10000.jsonl"],
        "grpo": [ROOT / "outputs/phase3a_grpo_pilot/preflight/train_subset_2000.jsonl",
                 ROOT / "outputs/phase4b_online_uvfc/data/train_10000.jsonl"],
    }
    stage_sets = {name: {image_basename(r) for p in paths for r in read_jsonl(p)}
                  for name, paths in stages.items()}
    for name, members in stage_sets.items():
        if not members <= {os.path.basename(r["image_path"]) for r in train}:
            raise AssertionError(f"{name} contains images outside the SFT train split")

    rows = []
    for row in train:
        base = os.path.basename(row["image_path"])
        used_by = ["sft"] + [name for name, members in stage_sets.items() if base in members]
        rows.append({
            "canonical_image_key": f"coco2014:{row['image_id']}",
            "source_dataset": "VQAv2",
            "source_split": "train2014",
            "image_path": row["image_path"],
            "image_id": str(row["image_id"]),
            "used_by": used_by,
        })
    return rows


def load_gqa_split(name: str) -> dict[str, list[str]]:
    questions = json.loads((GQA_QUESTIONS / name).read_text())
    refs: dict[str, list[str]] = collections.defaultdict(list)
    for record in questions.values():
        if record["isBalanced"]:
            refs[record["imageId"]].append(record["question"])
    return {image_id: sorted(set(texts)) for image_id, texts in refs.items()}


def audit_candidates(candidates: dict[str, list[str]], project: dict[str, dict],
                     project_rel: dict[str, str]) -> tuple[dict, list[dict], set[str]]:
    cand_ids = sorted(candidates)
    cand_rel = {i: f"data/raw/gqa/images_pool/{i}.jpg" for i in cand_ids}

    project_sha = {info["file_sha256"]: rel for rel, info in project.items()}
    project_rgb = {info["rgb_sha256"]: rel for rel, info in project.items()}
    proj_keys = list(project)
    proj_phash = np.array([project[k]["phash"] for k in proj_keys], dtype=np.uint64)

    cand_hash = _hash_many([cand_rel[i] for i in cand_ids])

    exclusions: list[dict] = []
    excluded: set[str] = set()
    flagged: set[str] = set()
    id_overlap_train = id_overlap_dev = 0
    train_ids = {r["image_id"] for r in read_jsonl(SFT_TRAIN)}
    dev_ids = {str(r["image_id"]) for r in read_jsonl(DEV_MANIFEST)}

    for wid in cand_ids:
        if str(wid) in train_ids:
            id_overlap_train += 1
        if str(wid) in dev_ids:
            id_overlap_dev += 1
        info = cand_hash[cand_rel[wid]]
        for reason, table, score in (("sha256", project_sha, info["file_sha256"]),
                                     ("rgb_hash", project_rgb, info["rgb_sha256"])):
            if score in table:
                exclusions.append({"gqa_image_id": wid, "project_image_id": table[score],
                                   "reason": reason, "distance_or_score": score})
                excluded.add(wid)
                break

    # near-duplicate: candidate-vs-exposure pHash hamming <= 4, then second-level verify
    cand_phash = {i: cand_hash[cand_rel[i]]["phash"] for i in cand_ids}
    near_pairs: list[tuple[str, str, int]] = []
    step = 2000
    for start in range(0, len(cand_ids), step):
        block = cand_ids[start:start + step]
        arr = np.array([cand_phash[i] for i in block], dtype=np.uint64)
        dist = np.bitwise_count(arr[:, None] ^ proj_phash[None, :])
        for gi, pi in np.argwhere(dist <= NEAR_DUP_MAX_DISTANCE):
            near_pairs.append((block[int(gi)], proj_keys[int(pi)], int(dist[gi, pi])))

    for wid, pkey, distance in near_pairs:
        flagged.add(wid)
        if wid in excluded:
            continue
        check = _second_level_check((cand_rel[wid], pkey))
        if check["corr"] >= SECOND_LEVEL_CORR:
            exclusions.append({"gqa_image_id": wid, "project_image_id": project_rel[pkey],
                               "reason": "near_duplicate", "distance_or_score": distance, **check})
            excluded.add(wid)

    stats = {
        "gqa_candidate_images": len(cand_ids),
        "train_exact_id_overlap": id_overlap_train,
        "dev_exact_id_overlap": id_overlap_dev,
        "exact_file_hash_overlap": sum(1 for e in exclusions if e["reason"] == "sha256"),
        "normalized_rgb_overlap": sum(1 for e in exclusions if e["reason"] == "rgb_hash"),
        "near_duplicate_flagged": len(flagged),
        "near_duplicate_excluded": sum(1 for e in exclusions if e["reason"] == "near_duplicate"),
        "remaining_images": len(cand_ids) - len(excluded),
    }
    return stats, exclusions, excluded


def reference_stats(selected: list[str], candidates: dict[str, list[str]],
                    typed: dict[str, dict] | None) -> dict:
    counts = [len(candidates[i]) for i in selected]
    out = {
        "images": len(selected),
        "total_reference_questions": int(sum(counts)),
        "mean_refs_per_image": round(statistics.mean(counts), 4),
        "median_refs_per_image": statistics.median(counts),
        "min_refs_per_image": min(counts),
        "max_refs_per_image": max(counts),
    }
    if typed is not None:
        structural = collections.Counter()
        semantic = collections.Counter()
        for i in selected:
            for rec in typed[i]:
                structural[rec["types"]["structural"]] += 1
                semantic[rec["types"]["semantic"]] += 1
        out["structural_type_distribution"] = dict(sorted(structural.items()))
        out["semantic_type_distribution"] = dict(sorted(semantic.items()))
    else:
        out["structural_type_distribution"] = "unavailable (GQA test annotations hidden)"
        out["semantic_type_distribution"] = "unavailable (GQA test annotations hidden)"
    return out


def sample(clean: list[str], n: int) -> list[str]:
    ordered = sorted(clean, key=lambda x: int(x[1:]) if x.startswith("n") else int(x))
    if len(ordered) < n:
        raise AssertionError(f"only {len(ordered)} clean candidates, need {n}")
    return sorted(random.Random(SEED).sample(ordered, n))


def main() -> int:
    started = time.time()
    for rel, expected in KNOWN_SHA256.items():
        actual = sha256_file(ROOT / rel)
        if actual != expected:
            raise AssertionError(f"protected input changed: {rel}\n{actual} != {expected}")

    train_rows = build_train_exposure()
    train_files = {os.path.basename(r["image_path"]) for r in train_rows}
    dev_rows = read_jsonl(DEV_MANIFEST)
    dev_files = {os.path.basename(r["image_path"]) for r in dev_rows}
    if train_files & dev_files:
        raise AssertionError("train exposure overlaps development1000")

    val_refs = load_gqa_split("val_balanced_questions.json")
    test_refs = load_gqa_split("test_balanced_questions.json")
    val_questions = json.loads((GQA_QUESTIONS / "val_balanced_questions.json").read_text())
    val_typed = collections.defaultdict(list)
    for rec in val_questions.values():
        if rec["isBalanced"]:
            val_typed[rec["imageId"]].append(rec)

    project = {}
    project_rel = {}
    for rows, role in ((train_rows, "train"), (dev_rows, "dev")):
        for r in rows:
            rel = r["image_path"]
            project[rel] = None
            project_rel[rel] = f"{role}:{r['image_id']}"

    hashed = _hash_many(sorted(project))
    for rel in project:
        project[rel] = hashed[rel]

    audit = {"schema": "gqa-cross-dataset-overlap-audit-v1",
             "source_id_mapping": ("unavailable: GQA imageId is a Visual Genome id, not a COCO id; "
                                   "numeric collisions with project COCO ids were verified by exact and "
                                   "normalized-RGB hashing to be different images"),
             "near_duplicate_max_hamming": NEAR_DUP_MAX_DISTANCE,
             "second_level_corr_threshold": SECOND_LEVEL_CORR}
    all_exclusions = []
    excluded_by_split = {}
    for split_name, refs in (("test_balanced", test_refs), ("val_balanced", val_refs)):
        stats, exclusions, excluded = audit_candidates(refs, project, project_rel)
        audit[split_name] = stats
        all_exclusions.extend(exclusions)
        excluded_by_split[split_name] = excluded

    clean = {}
    for split_name, refs in (("test_balanced", test_refs), ("val_balanced", val_refs)):
        clean[split_name] = [i for i in sorted(refs) if i not in excluded_by_split[split_name]]

    gen_ids = sample(clean["test_balanced"], GEN_N)
    sanity_pool = [i for i in clean["val_balanced"] if i not in set(gen_ids)]
    sanity_ids = sample(sanity_pool, SANITY_N)

    # ---- integrity asserts -------------------------------------------------
    assert len(gen_ids) == GEN_N and len(set(gen_ids)) == GEN_N
    assert len(sanity_ids) == SANITY_N and len(set(sanity_ids)) == SANITY_N
    assert not (set(gen_ids) & set(sanity_ids))
    assert not (set(gen_ids) & excluded_by_split["test_balanced"])
    for iid in gen_ids + sanity_ids:
        path = GQA_IMAGES / f"{iid}.jpg"
        assert path.exists(), f"missing image {iid}"
        with Image.open(path) as im:
            im.verify()
    assert all(len(test_refs[i]) >= 1 for i in gen_ids)
    assert all(len(val_refs[i]) >= 1 for i in sanity_ids)
    seen_ids = {str(r["image_id"]) for r in train_rows} | {str(r["image_id"]) for r in dev_rows}
    assert not (set(gen_ids) & seen_ids)
    assert not (set(sanity_ids) & seen_ids)

    # ---- write outputs -----------------------------------------------------
    write_jsonl(OUT / "train_exposure_images.jsonl", train_rows)
    write_jsonl(OUT / "gqa_generalization1000_images.jsonl",
                [{"image_id": i, "image_path": f"data/raw/gqa/images_pool/{i}.jpg",
                  "source": "GQA", "split": "test_balanced"} for i in gen_ids])
    write_jsonl(OUT / "gqa_sanity50_images.jsonl",
                [{"image_id": i, "image_path": f"data/raw/gqa/images_pool/{i}.jpg",
                  "source": "GQA", "split": "val_balanced"} for i in sanity_ids])
    (OUT / "gqa_generalization1000_refs.json").write_text(json.dumps(
        {i: {"references": test_refs[i]} for i in gen_ids}, ensure_ascii=False, indent=1))
    (OUT / "gqa_sanity50_refs.json").write_text(json.dumps(
        {i: {"references": val_refs[i]} for i in sanity_ids}, ensure_ascii=False, indent=1))

    audit["totals"] = {
        "gqa_candidate_images": len(test_refs) + len(val_refs),
        "excluded": len(all_exclusions),
        "remaining_images": len(clean["test_balanced"]) + len(clean["val_balanced"]),
    }
    (OUT / "gqa_overlap_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2))
    write_jsonl(OUT / "gqa_overlap_exclusions.jsonl", all_exclusions)

    ref_stats = {
        "generalization1000": reference_stats(gen_ids, test_refs, None),
        "sanity50": reference_stats(sanity_ids, val_refs, val_typed),
    }

    # ---- source + contract manifests --------------------------------------
    def qfile(name: str) -> dict:
        p = GQA_QUESTIONS / name
        return {"local_path": str(p.relative_to(ROOT)), "size_bytes": p.stat().st_size,
                "sha256": sha256_file(p)}
    image_zip = GQA_ARCHIVES / "images.zip"
    source = {
        "dataset": "GQA", "version": "1.2",
        "source_url": {
            "images": "https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip",
            "questions": "https://downloads.cs.stanford.edu/nlp/data/gqa/questions1.2.zip",
        },
        "download_date_note": "derived from archive mtime; downloader log empty",
        "images": {"local_pool": "data/raw/gqa/images_pool",
                   "image_count": len(os.listdir(GQA_IMAGES)),
                   "archive": "data/raw/archives/gqa/images.zip",
                   "archive_size_bytes": image_zip.stat().st_size,
                   "archive_sha256": sha256_file(image_zip),
                   "download_mtime": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(image_zip.stat().st_mtime))},
        "questions1.2": {
            "archive": "data/raw/archives/gqa/questions1.2.zip",
            "archive_sha256": sha256_file(GQA_ARCHIVES / "questions1.2.zip"),
            "test_balanced_questions.json": qfile("test_balanced_questions.json"),
            "val_balanced_questions.json": qfile("val_balanced_questions.json"),
        },
        "not_used_as_model_input": ["sceneGraphs", "object features", "spatial features",
                                    "functional programs", "answers", "GQA questions"],
        "reference_only": ["test_balanced_questions.json", "val_balanced_questions.json"],
    }
    (OUT / "gqa_source_manifest.json").write_text(json.dumps(source, ensure_ascii=False, indent=2))

    def describe(path: Path, count_key: str) -> dict:
        rel = str(path.relative_to(ROOT))
        rows = read_jsonl(path)
        entry = {"path": rel, "records": len(rows), "sha256": sha256_file(path)}
        if rows and count_key in rows[0]:
            entry["unique_images"] = len({str(r[count_key]) for r in rows})
        return entry

    refs_gen = json.loads((OUT / "gqa_generalization1000_refs.json").read_text())
    refs_san = json.loads((OUT / "gqa_sanity50_refs.json").read_text())
    manifest = {
        "schema": "cross-dataset-data-contract-v1",
        "phase": "G0",
        "seed": SEED,
        "creation_script": "scripts/vvqg/freeze_data_contract.py",
        "git_commit": os.popen("git rev-parse HEAD").read().strip(),
        "train_exposure": {
            "manifest": "src/vvqg/data/sft/train_manifest.jsonl",
            "records": len(train_rows),
            "unique_images": len(train_files),
            "used_by_counts": {k: sum(k in r["used_by"] for r in train_rows)
                               for k in ("sft", "uvfc", "teacher_generation", "distillation", "grpo")},
        },
        "development1000": {"manifest": "outputs/same_backbone_baselines/manifests/development1000_images.jsonl",
                            "records": len(dev_rows), "unique_images": len(dev_files), "role": "IN_DOMAIN_DEV",
                            "note": "reward-calibration corpus is the same 1000 images (method selection, not weights)"},
        "files": [
            describe(OUT / "train_exposure_images.jsonl", "image_id"),
            {"path": "outputs/same_backbone_baselines/manifests/development1000_images.jsonl",
             "records": len(dev_rows), "unique_images": len(dev_files), "sha256": sha256_file(DEV_MANIFEST)},
            describe(OUT / "gqa_generalization1000_images.jsonl", "image_id"),
            describe(OUT / "gqa_sanity50_images.jsonl", "image_id"),
            {"path": "outputs/data_contract/gqa_generalization1000_refs.json",
             "records": len(refs_gen), "unique_images": len(refs_gen),
             "sha256": sha256_file(OUT / "gqa_generalization1000_refs.json")},
            {"path": "outputs/data_contract/gqa_sanity50_refs.json",
             "records": len(refs_san), "unique_images": len(refs_san),
             "sha256": sha256_file(OUT / "gqa_sanity50_refs.json")},
            {"path": "outputs/data_contract/gqa_overlap_audit.json", "sha256": sha256_file(OUT / "gqa_overlap_audit.json")},
            describe(OUT / "gqa_overlap_exclusions.jsonl", "gqa_image_id"),
            {"path": "outputs/data_contract/gqa_source_manifest.json", "sha256": sha256_file(OUT / "gqa_source_manifest.json")},
        ],
        "reference_stats": ref_stats,
        "integrity": {
            "generalization_images": len(gen_ids),
            "generalization_unique": len(set(gen_ids)),
            "sanity_images": len(sanity_ids),
            "sanity_unique": len(set(sanity_ids)),
            "excluded_test": len(excluded_by_split["test_balanced"]),
            "excluded_val": len(excluded_by_split["val_balanced"]),
            "generation_manifests_question_free": True,
        },
    }
    (OUT / "data_contract_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))

    print(f"DATA CONTRACT FROZEN in {time.time() - started:.1f}s")
    print(json.dumps(manifest["integrity"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
