"""Download-on-demand fetcher for benchmark network data (P9).

TNTP data (github.com/bstabler/TransportationNetworks) is donated for
academic research and carries no OSI license, so TABenchmark never vendors
it. Files are fetched from a **commit-pinned** raw URL, verified against
SHA-256 checksums recorded in the registry, cached locally, and cited.
"""

from __future__ import annotations

import hashlib
import os
import urllib.request
from pathlib import Path

from .registry import NetworkSpec

__all__ = ["ChecksumError", "cache_dir", "fetch", "citation"]

TNTP_COMMIT = "d1639b4ef218c17928ba573e806ddf8ba5e7ae6d"
TNTP_BASE = f"https://raw.githubusercontent.com/bstabler/TransportationNetworks/{TNTP_COMMIT}"


class ChecksumError(RuntimeError):
    """Downloaded file does not match the pinned checksum."""


def cache_dir() -> Path:
    """Local cache root: ``$TABENCH_CACHE`` or ``~/.cache/tabench``."""
    root = os.environ.get("TABENCH_CACHE")
    path = Path(root) if root else Path.home() / ".cache" / "tabench"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(spec: NetworkSpec, force: bool = False, timeout: float = 60.0) -> dict[str, Path]:
    """Ensure all files for a network are cached and checksum-verified.

    Returns a mapping ``role -> local path`` (roles: ``net``, ``trips``,
    ``flow``, ``node`` as available).
    """
    target_dir = cache_dir() / spec.key
    target_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for role, (filename, checksum) in spec.files.items():
        local = target_dir / filename
        if force or not local.exists():
            url = f"{TNTP_BASE}/{spec.repo_dir}/{filename}"
            tmp = local.with_suffix(local.suffix + ".part")
            with urllib.request.urlopen(url, timeout=timeout) as resp, open(tmp, "wb") as out:
                out.write(resp.read())
            tmp.replace(local)
        actual = _sha256(local)
        if actual != checksum:
            local.unlink(missing_ok=True)
            raise ChecksumError(
                f"{spec.key}/{filename}: checksum mismatch "
                f"(expected {checksum[:12]}…, got {actual[:12]}…). "
                "File removed from cache; re-run to re-download."
            )
        paths[role] = local
    return paths


def citation(spec: NetworkSpec) -> str:
    """Mandatory citation string for the data source."""
    base = (
        "Transportation Networks for Research Core Team. Transportation Networks "
        "for Research. https://github.com/bstabler/TransportationNetworks "
        f"(commit {TNTP_COMMIT[:12]})."
    )
    if spec.citation_extra:
        return f"{base} {spec.citation_extra}"
    return base
