"""Local model locations; explicit CLI/config paths take precedence."""

import os
from pathlib import Path


def local_model_path(name: str) -> str:
    return str(Path(os.environ.get("VVQG_MODELS_DIR", Path.home() / "models")).expanduser() / name)
