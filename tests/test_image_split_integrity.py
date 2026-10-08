"""Phase-7 tests for the immutable image-split manifest and the train-only gate."""
from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

import pytest

from vqg import image_splits as S

EXPECTED_COUNTS = {
    "gqa": {"train": 72140, "val": 10234, "test": 2987, "testdev": 398},
    "vqav2": {"train": 82783, "val": 40504},
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


@pytest.mark.integration
@pytest.mark.external_artifact("data/train_resources/image_split_manifest.json")
class ManifestTest(unittest.TestCase):
    def setUp(self) -> None:
        self.manifest = S.load_manifest()

    def test_schema_and_manifest_path_resolution(self) -> None:
        self.assertEqual(self.manifest["schema_version"], "image-split-manifest-1.0")
        self.assertTrue(S.MANIFEST_PATH.is_file())
        self.assertEqual(S.MANIFEST_PATH.resolve(),
                         (S.REPO_ROOT / "data/train_resources/image_split_manifest.json").resolve())

    def test_counts_and_official_split_names(self) -> None:
        for dataset, splits in EXPECTED_COUNTS.items():
            for split, expected in splits.items():
                ids = S.image_ids(dataset, split)
                self.assertEqual(len(ids), expected, f"{dataset}/{split}")
                payload = self.manifest["datasets"][dataset]["splits"][split]
                self.assertTrue(payload["official_split"])

    def test_image_id_list_hashes_are_consistent(self) -> None:
        for dataset, payload in self.manifest["datasets"].items():
            for split, entry in payload["splits"].items():
                text = "\n".join(sorted(set(entry["image_ids"])))
                if text:
                    text += "\n"
                digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
                self.assertEqual(digest, entry["image_ids_sha256"],
                                 f"{dataset}/{split} image_ids_sha256")

    def test_source_sha256_matches_raw_file(self) -> None:
        entry = self.manifest["datasets"]["gqa"]["splits"]["test"]
        if not (S.REPO_ROOT / entry["source_path"]).is_file():
            pytest.skip("requires external artifact: " + entry["source_path"])
        self.assertEqual(_sha256_file(S.REPO_ROOT / entry["source_path"]),
                         entry["source_sha256"])

    def test_gqa_train_and_eval_are_nonempty_and_disjoint(self) -> None:
        train = S.train_image_ids("gqa")
        evaluation = S.eval_image_ids("gqa")
        self.assertGreater(len(train), 0)
        self.assertGreater(len(evaluation), 0)
        self.assertEqual(len(train & evaluation), 0)
        self.assertTrue(S.assert_disjoint(train, evaluation, "gqa train/eval"))


class TrainOnlyGateTest(unittest.TestCase):
    @pytest.fixture(autouse=True)
    def synthetic_manifest(self, tmp_path, monkeypatch):
        path = tmp_path / "splits.json"
        path.write_text(json.dumps({"datasets": {
            dataset: {"eval_splits": ["val", "test"], "splits": {
                "train": {"image_ids": ["train1", "train2"]},
                "val": {"image_ids": ["eval1"]},
                "test": {"image_ids": ["eval2"]},
            }} for dataset in ("gqa", "vqav2")
        }}))
        monkeypatch.setattr(S, "MANIFEST_PATH", path)

    def _rows(self) -> list[dict]:
        train_id = sorted(S.train_image_ids("gqa"))[0]
        eval_id = sorted(S.eval_image_ids("gqa"))[0]
        return [
            {"pair_id": "t", "image_id": train_id, "source_split": "train"},
            {"pair_id": "v", "image_id": eval_id, "source_split": "val"},
            {"pair_id": "te", "image_id": eval_id, "source_split": "test"},
            {"pair_id": "t-eval-img", "image_id": eval_id, "source_split": "train"},
        ]

    def test_filter_drops_injected_val_and_test_rows(self) -> None:
        rows = self._rows()
        kept, audit = S.filter_train_only(rows, "gqa")
        self.assertEqual([r["pair_id"] for r in kept], ["t"])
        self.assertEqual(audit["input_rows"], 4)
        self.assertEqual(audit["kept_rows"], 1)
        self.assertEqual(audit["dropped_rows"], 3)
        self.assertEqual(audit["eval_overlap_before"], 3)
        self.assertEqual(audit["eval_overlap_after"], 0)
        self.assertEqual(audit["dropped_by_split"],
                         {"test": 1, "val": 1, "eval_overlap": 1})

    def test_filter_never_mutates_input(self) -> None:
        rows = self._rows()
        before = json.dumps(rows, sort_keys=True)
        S.filter_train_only(rows, "gqa")
        self.assertEqual(json.dumps(rows, sort_keys=True), before)

    def test_assert_no_eval_overlap_raises_on_eval_image(self) -> None:
        eval_id = sorted(S.eval_image_ids("gqa"))[0]
        S.assert_no_eval_overlap(sorted(S.train_image_ids("gqa"))[:3], "gqa", "ok")
        with self.assertRaises(ValueError) as ctx:
            S.assert_no_eval_overlap([eval_id], "gqa", "injected")
        self.assertIn("1 image ids overlap", str(ctx.exception))

    def test_vqav2_train_rows_are_kept(self) -> None:
        train_id = sorted(S.train_image_ids("vqav2"))[0]
        eval_id = sorted(S.eval_image_ids("vqav2"))[0]
        rows = [
            {"image_id": train_id, "source_split": "train"},
            {"image_id": eval_id, "source_split": "val"},
        ]
        kept, audit = S.filter_train_only(rows, "vqav2")
        self.assertEqual(len(kept), 1)
        self.assertEqual(audit["eval_overlap_before"], 1)
        self.assertEqual(audit["eval_overlap_after"], 0)

    def test_assert_disjoint_reports_overlap(self) -> None:
        with self.assertRaises(ValueError):
            S.assert_disjoint(["a", "b"], ["b", "c"], "demo")

    def test_summarize_rows_flags_eval_overlap(self) -> None:
        summary = S.summarize_rows(self._rows(), "gqa")
        self.assertEqual(summary["input_rows"], 4)
        self.assertEqual(summary["unique_images"], 2)
        self.assertEqual(summary["eval_overlap_rows"], 3)
        self.assertEqual(summary["eval_overlap_images"], 1)


if __name__ == "__main__":
    unittest.main()
