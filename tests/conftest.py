"""Shared test helpers.

``load_or_skip`` gates network-backed fixtures: offline contributors get a
skip, but checksum mismatches always fail, and setting TABENCH_REQUIRE_DATA=1
(as CI does) turns ANY data failure — 404s, timeouts, parser regressions —
into a hard error so a broken data pipeline can never hide behind green skips.
"""

import os

import pytest

from tabench.data import ChecksumError, load_scenario


def load_or_skip(key: str):
    try:
        return load_scenario(key)
    except ChecksumError:
        raise
    except Exception as exc:
        if os.environ.get("TABENCH_REQUIRE_DATA"):
            raise
        pytest.skip(f"{key} data unavailable: {exc}")
