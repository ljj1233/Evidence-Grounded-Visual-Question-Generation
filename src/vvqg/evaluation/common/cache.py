"""Shared cache-resume policy for API evaluators.

Both the SeeQ (GPT-5.6-Luna) and visual-support (GPT-4o) evaluators persist a
row per content-addressed cache key. A key present in the cache is reused; with
``retry_failures`` a cached non-``ok`` row is requested again. Failed requests
are never treated as a score: they are stored without scores and excluded from
the denominator.
"""

from __future__ import annotations


def needs_request(cache_key: str, cached: dict, retry_failures: bool = False) -> bool:
    if cache_key not in cached:
        return True
    return bool(retry_failures and cached[cache_key].get("status") != "ok")
