import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vvqg.generation import GenerationConfig, VQGGenerator

spec = importlib.util.spec_from_file_location("generation_cli", Path(__file__).resolve().parents[1] / "scripts/vvqg/generate.py")
cli = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cli)


def test_resume_keeps_full_original_batches_and_sample_positions():
    ids = list("abcdef")
    done = {(iid, g) for iid in ids[:3] for g in range(4)} | {("d", 0), ("d", 1)}
    assert list(cli.pending_batches(ids, done, 2, 4)) == [["c", "d"], ["e", "f"]]
    assert list(cli.pending_batches(ids, {(iid, g) for iid in ids for g in range(4)}, 2, 4)) == []


def test_zero_shot_cli_excludes_checkpoint():
    args = cli.parse_args(["--data", "in.jsonl", "--output", "out.jsonl", "--zero-shot"])
    assert args.zero_shot
    with pytest.raises(SystemExit):
        cli.parse_args(["--data", "in.jsonl", "--output", "out.jsonl", "--zero-shot", "--checkpoint", "adapter"])


def test_oom_halves_batch_preserves_order_and_never_swallows_other_errors():
    class OOM(RuntimeError):
        pass

    gen = VQGGenerator.__new__(VQGGenerator)
    gen.config = GenerationConfig("base", batch_size=4)
    gen.oom_splits = 0
    gen.torch = SimpleNamespace(OutOfMemoryError=OOM, cuda=SimpleNamespace(empty_cache=lambda: None))
    calls = []

    def generate(paths, key):
        calls.append((paths, key))
        if len(paths) > 2:
            raise OOM()
        return [[path] for path in paths]

    gen.generate_batch = generate
    items = [(str(i), f"image{i}") for i in range(4)]
    assert gen.generate_many(items) == [(iid, [path]) for iid, path in items]
    assert [key for _, key in calls] == ["0,1,2,3", "0,1", "2,3"]
    assert gen.oom_splits == 1
    gen.generate_batch = lambda *_: (_ for _ in ()).throw(ValueError("real bug"))
    with pytest.raises(ValueError, match="real bug"):
        gen.generate_many(items)
    gen.generate_batch = lambda *_: (_ for _ in ()).throw(OOM())
    with pytest.raises(OOM):
        gen.generate_many(items[:1])
