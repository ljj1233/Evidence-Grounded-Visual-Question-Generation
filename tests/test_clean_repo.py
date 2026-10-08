import ast
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from vvqg.paths import local_model_path

ROOT = Path(__file__).resolve().parents[1]


def test_model_directory_override(monkeypatch, tmp_path):
    monkeypatch.setenv("VVQG_MODELS_DIR", str(tmp_path / "models with spaces"))
    assert local_model_path("Qwen3-VL-8B-Instruct") == str(tmp_path / "models with spaces/Qwen3-VL-8B-Instruct")
    monkeypatch.setenv("VVQG_MODELS_DIR", "~/custom-models")
    assert local_model_path("Qwen3-Embedding-4B") == str(Path.home() / "custom-models/Qwen3-Embedding-4B")


def test_configs_and_schemas_load_without_external_assets():
    for path in (ROOT / "configs").rglob("*.yaml"):
        assert isinstance(yaml.safe_load(path.read_text()), dict), path
    for path in (ROOT / "src/vvqg/data").rglob("*schema.json"):
        schema = json.loads(path.read_text())
        assert schema["type"] == "object", path
        assert isinstance(schema["properties"], dict), path


def test_current_runtime_has_no_private_absolute_paths():
    for directory in ("src", "shared", "scripts/vvqg"):
        for path in (ROOT / directory).rglob("*.py"):
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.Constant) and isinstance(node.value, str):
                    assert not node.value.startswith(("/hpc", "/mnt/")), (path, node.lineno)


@pytest.mark.parametrize("entrypoint", [
    "src/vvqg/training/train_sft.py",
    "src/vvqg/training/train_grpo.py",
    "scripts/vvqg/dpo_v1_formal_contract.py",
    "scripts/vvqg/generate.py",
])
def test_cli_help_without_pythonpath(entrypoint):
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    result = subprocess.run([sys.executable, "-B", str(ROOT / entrypoint), "--help"],
                            cwd=ROOT, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout
