"""Evidence-grounded educational visual question generation."""

import os
import sys

__version__ = "0.1.0"

DEFAULT_NUM_THREADS = 8
_THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


def configure_threads(n: int | None = None) -> int:
    """Cap BLAS/OpenMP/torch at ``n`` threads (default 8; override with ``EGVQG_NUM_THREADS``).

    Must run before numpy/torch import; this module executes first for any ``egvqg.*`` import.
    """
    if n is None:
        n = int(os.environ.get("EGVQG_NUM_THREADS", DEFAULT_NUM_THREADS))
    n = max(1, n)
    for key in _THREAD_ENV_VARS:
        os.environ[key] = str(n)
    torch = sys.modules.get("torch")
    if torch is not None:
        torch.set_num_threads(n)
        try:
            torch.set_num_interop_threads(n)
        except RuntimeError:
            pass
    return n


configure_threads()
