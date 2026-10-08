from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def require_external_artifacts(request):
    paths = {path for mark in request.node.iter_markers("external_artifact") for path in mark.args}
    missing = sorted(path for path in paths if not (ROOT / path).is_file())
    if missing:
        pytest.skip("requires external artifact: " + ", ".join(missing))
