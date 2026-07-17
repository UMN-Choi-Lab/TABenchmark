"""EDOC canonicalization: the versioned, hashed harness module (ruling R10).

Design: docs/design/adr-036-external-dynamic-observational-certificate.md.

The canonicalization spec is **versioned** (domain ``"tabench-edoc-canon-v1;"``)
so that an upstream engine-format drift bumps the version and mints new instance
hashes, disclosed. Its founding spec is the **four measured necessities** from the
pilots: strip the SUMO ``generated on`` timestamp comment + the ``summary``
wall-clock ``duration`` attribute; sort MATSim same-timestamp event ties; hash the
**decompressed** payload; positional-parse the DTALite trajectory. (This module
ships the SUMO canonicalizer — the first row, adr-037 — and the MATSim
canonicalizer — the second row, adr-039 — under the same version; the DTALite
canonicalizer lands with its row, S4.)

**MATSim tie-sort (adr-039, the corrected R10 record):** the same-timestamp
event-order permutation is a MULTITHREADING artifact, not a replay-vs-original
effect — measured: at ``numberOfThreads=8`` two identical-seed runs permute
104/1400 event lines within equal timestamps (multiset identical), while at the
pinned ``numberOfThreads=1`` the replay is raw-byte-identical to the certified
run even with forced ties. The canonicalizer sorts each same-timestamp run of
``<event .../>`` lines by their full line bytes (stable, content-sensitive,
idempotent), so the G1 hash is invariant across thread counts / replay seeds /
input order while any CONTENT difference still moves it; raw-byte identity at
threads=1 is a stricter engine-gated bonus check, never the gate.

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

# MATSim G1 hash surface (adr-039): an explicit ALLOWLIST, not the SUMO log
# denylist — measured by the twin-run byte census on the shipped family (the S2
# 23/184 analogue: 66/69 output files raw-identical across identical-seed runs
# in different tmpdirs; recorded in adr-039). The three divergent files —
# ``logfile.log``, ``logfileWarningsErrors.log``, ``stopwatch.csv`` — carry
# wall-clock text and are provenance, NEVER hashed (R10: simulation-state
# artifacts only). ``output_config.xml`` is on the surface: the certifier
# writes purely RELATIVE paths, so the engine's config echo is byte-stable
# across working dirs (census-confirmed) and hashing it pins the engine's own
# record of ``firstIteration == lastIteration`` (forgery pair N6).
_MATSIM_HASHED_BASENAMES = frozenset(
    {
        "output_events.xml.gz",
        "output_plans.xml.gz",
        "output_network.xml.gz",
        "output_config.xml",
    }
)

# The multi-line "<!-- generated on ... by Eclipse SUMO ... -->" header SUMO
# writes atop every XML output (it embeds the full run configuration, so it spans
# many lines up to the first "-->"). Non-greedy + DOTALL captures exactly it.
_SUMO_GENERATED_COMMENT = re.compile(rb"<!--(?:(?!-->).)*?generated on.*?-->\s*", re.DOTALL)
# Wall-clock per-step duration attr inside SUMO summary outputs.
_SUMO_DURATION_ATTR = re.compile(rb'\s+duration="[^"]*"')


def _basename(name: str) -> str:
    return name.rsplit("/", 1)[-1]


def is_hashed_artifact(name: str) -> bool:
    """SUMO hash surface (R10): simulation-state artifacts are hashed; engine/
    driver logs are provenance and excluded. ``name`` may be a path or a basename."""
    base = _basename(name)
    if base in _LOG_BASENAMES:
        return False
    return not base.endswith(_LOG_SUFFIXES)


def is_hashed_matsim_artifact(name: str) -> bool:
    """MATSim hash surface (R10, adr-039): allowlist-only — everything outside
    ``_MATSIM_HASHED_BASENAMES`` is provenance (census-measured; see the
    constant's comment). ``name`` may be a path or a basename."""
    return _basename(name) in _MATSIM_HASHED_BASENAMES


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


# The multi-line MATSim event element; the permutation surface is exactly runs of
# consecutive ``<event .../>`` lines sharing one ``time="..."`` attribute value.
_MATSIM_EVENT_TIME = re.compile(rb'\btime="([^"]*)"')


def _sort_matsim_event_ties(payload: bytes) -> bytes:
    """Stable-sort each run of consecutive same-timestamp ``<event .../>`` lines
    by their full line bytes (R10 tie canonicalization, adr-039). Header/footer
    and any non-event line pass through untouched and delimit the runs. Sorting
    by the complete line is content-sensitive (a doctored attribute still moves
    the hash) and idempotent (a sorted run re-sorts to itself)."""
    lines = payload.split(b"\n")
    out: list[bytes] = []
    run: list[bytes] = []
    run_time: bytes | None = None

    def _flush() -> None:
        nonlocal run, run_time
        if run:
            out.extend(sorted(run))
        run, run_time = [], None

    for line in lines:
        if b"<event " in line:
            m = _MATSIM_EVENT_TIME.search(line)
            t = m.group(1) if m is not None else None
            if t is not None and t == run_time:
                run.append(line)
                continue
            _flush()
            run_time = t
            run = [line]
        else:
            _flush()
            out.append(line)
    _flush()
    return b"\n".join(out)


def canonicalize_matsim(name: str, data: bytes) -> bytes:
    """Canonicalize one MATSim artifact (adr-039): decompress (gzip frames carry
    volatile metadata), and — for events artifacts ONLY — sort same-timestamp
    event ties (the measured multithreading permutation surface; see the module
    docstring). Plans/network payloads pass through decompressed and untouched.
    Idempotent."""
    out = decompress(data)
    if "events" in _basename(name):
        out = _sort_matsim_event_ties(out)
    return out


def hash_artifacts(
    artifacts: Mapping[str, bytes],
    canonicalizer: Callable[[str, bytes], bytes],
    surface: Callable[[str], bool] = is_hashed_artifact,
) -> str:
    """SHA-256 over the canonicalized hash-surface artifacts, domain-separated
    (``"tabench-edoc-canon-v1;"``) and per-artifact LENGTH-FRAMED (``name:size;``
    then the canonical bytes), artifacts taken in sorted name order so the digest
    is order-independent. Log/provenance artifacts are excluded by the engine's
    ``surface`` predicate (R10 hash surface; the default is the SUMO surface, so
    every pre-S3 digest is byte-identical — regression-pinned in test_edoc).
    This is the object G1's replay-fidelity determinism double compares.
    """
    h = hashlib.sha256()
    h.update(_CANON_DOMAIN)
    for name in sorted(artifacts):
        if not surface(name):
            continue
        canon = canonicalizer(name, artifacts[name])
        h.update(f"{name}:{len(canon)};".encode())
        h.update(canon)
    return h.hexdigest()


def hash_sumo_artifacts(artifacts: Mapping[str, bytes]) -> str:
    """SUMO specialization of :func:`hash_artifacts` (adr-037's first row)."""
    return hash_artifacts(artifacts, canonicalize_sumo)


def hash_matsim_artifacts(artifacts: Mapping[str, bytes]) -> str:
    """MATSim specialization of :func:`hash_artifacts` (adr-039's row): the
    allowlist surface + the tie-sorting canonicalizer, same domain/version."""
    return hash_artifacts(artifacts, canonicalize_matsim, surface=is_hashed_matsim_artifact)
