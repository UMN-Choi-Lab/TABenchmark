"""EDOC canonicalization: the versioned, hashed harness module (ruling R10).

Design: docs/design/adr-036-external-dynamic-observational-certificate.md.

The canonicalization spec is **versioned** (domain ``"tabench-edoc-canon-v1;"``)
so that an upstream engine-format drift bumps the version and mints new instance
hashes, disclosed. Its founding spec is the **four measured necessities** from the
pilots: strip the SUMO ``generated on`` timestamp comment + the ``summary``
wall-clock ``duration`` attribute; sort MATSim same-timestamp event ties; hash the
**decompressed** payload; positional-parse the DTALite trajectory. (This module
ships the SUMO canonicalizer — the first row, adr-037; the MATSim/DTALite
canonicalizers land with their rows under the same version, S3/S4.)

It also defines an explicit **hash surface**: the strips do NOT make the whole
emitted tree byte-identical (measured: 23/184 SUMO files — ``*.sumo.log``,
``driver.out``, ``dua.log``, ``stdout.log`` — still differ on wall-clock text
after the two strips), so **only simulation-state artifacts are hashed** and
engine/driver logs are provenance, **never** on the G1 hash surface — else the
G1 determinism double would over-censor every honest run.

CRITICAL: the ``duration`` attribute is wall-clock only inside SUMO ``summary``
outputs (``<step ... duration="0"/>``); in ``tripinfo`` ``duration`` is the REAL
experienced trip cost and MUST NOT be stripped. The strip is therefore
name-scoped to summary artifacts.
"""

from __future__ import annotations

import gzip
import hashlib
import re
from collections.abc import Callable, Mapping

CANON_VERSION = "tabench-edoc-canon-v1"
_CANON_DOMAIN = b"tabench-edoc-canon-v1;"

# Engine/driver logs: wall-clock text, provenance only, NEVER hashed (R10).
_LOG_BASENAMES = frozenset({"driver.out", "dua.log", "stdout.log"})
_LOG_SUFFIXES = (".sumo.log",)

# The multi-line "<!-- generated on ... by Eclipse SUMO ... -->" header SUMO
# writes atop every XML output (it embeds the full run configuration, so it spans
# many lines up to the first "-->"). Non-greedy + DOTALL captures exactly it.
_SUMO_GENERATED_COMMENT = re.compile(rb"<!--(?:(?!-->).)*?generated on.*?-->\s*", re.DOTALL)
# Wall-clock per-step duration attr inside SUMO summary outputs.
_SUMO_DURATION_ATTR = re.compile(rb'\s+duration="[^"]*"')


def _basename(name: str) -> str:
    return name.rsplit("/", 1)[-1]


def is_hashed_artifact(name: str) -> bool:
    """Hash surface (R10): simulation-state artifacts are hashed; engine/driver
    logs are provenance and excluded. ``name`` may be a path or a basename."""
    base = _basename(name)
    if base in _LOG_BASENAMES:
        return False
    return not base.endswith(_LOG_SUFFIXES)


def _is_summary(name: str) -> bool:
    return "summary" in _basename(name)


def decompress(data: bytes) -> bytes:
    """Hash the DECOMPRESSED payload (R10): gzip frames carry volatile metadata
    (mtime byte, OS byte), so a ``.gz`` artifact is decompressed before any
    canonicalization or hashing. A non-gzip payload passes through unchanged."""
    if len(data) >= 2 and data[0] == 0x1F and data[1] == 0x8B:
        return gzip.decompress(data)
    return data


def canonicalize_sumo(name: str, data: bytes) -> bytes:
    """Canonicalize one SUMO artifact: decompress, strip the ``generated on``
    timestamp comment (all XML), and — for ``summary`` artifacts ONLY — strip the
    wall-clock ``duration`` attribute. Every simulation-state byte is left intact
    (tripinfo ``duration`` is untouched). Idempotent."""
    out = decompress(data)
    out = _SUMO_GENERATED_COMMENT.sub(b"", out)
    if _is_summary(name):
        out = _SUMO_DURATION_ATTR.sub(b"", out)
    return out


def hash_artifacts(
    artifacts: Mapping[str, bytes],
    canonicalizer: Callable[[str, bytes], bytes],
) -> str:
    """SHA-256 over the canonicalized hash-surface artifacts, domain-separated
    (``"tabench-edoc-canon-v1;"``) and per-artifact LENGTH-FRAMED (``name:size;``
    then the canonical bytes), artifacts taken in sorted name order so the digest
    is order-independent. Log/provenance artifacts are excluded (R10 hash
    surface). This is the object G1's replay-fidelity determinism double compares.
    """
    h = hashlib.sha256()
    h.update(_CANON_DOMAIN)
    for name in sorted(artifacts):
        if not is_hashed_artifact(name):
            continue
        canon = canonicalizer(name, artifacts[name])
        h.update(f"{name}:{len(canon)};".encode())
        h.update(canon)
    return h.hexdigest()


def hash_sumo_artifacts(artifacts: Mapping[str, bytes]) -> str:
    """SUMO specialization of :func:`hash_artifacts` (adr-037's first row)."""
    return hash_artifacts(artifacts, canonicalize_sumo)
