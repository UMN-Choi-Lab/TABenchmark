"""Enforcement for the per-unit tutorial notebooks (``tutorials/<NN>-<track>/<MM>-<unit>.ipynb``).

The PI rule "a new model must ship a tutorial" made mechanical:

* every key of ``MODEL_REGISTRY`` / ``ESTIMATOR_REGISTRY`` /
  ``DYNAMIC_ESTIMATOR_REGISTRY`` and every parallel-track unit (an import-anchored
  11-unit manifest — those tracks carry no registry) maps to a notebook, allowing
  same-ADR ``covers`` folds;
* the DNL track cannot grow silently — new ``LinkModel`` / ``NodeModel`` subclasses
  must appear in the manifest;
* notebooks are committed STRIPPED (no outputs, no execution counts) and their
  ``metadata.tabench`` block is folder-consistent;
* CI re-executes each one from a cleared state (gated on ``TABENCH_RUN_TUTORIALS=1``,
  mirroring the ``TABENCH_REQUIRE_DATA`` discipline in ``tests/conftest.py``).

The ``_ALLOWLIST`` of not-yet-written units SHRINKS batch by batch; when it empties
the coverage gate is fully strict. Guarded units (torch/sumo/dtalite) register only
where their extra is installed, so the enumeration is automatically environment-correct
and their notebooks are demanded in the extras CI jobs.
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
from pathlib import Path

import pytest

from tabench.estimation import DYNAMIC_ESTIMATOR_REGISTRY, ESTIMATOR_REGISTRY
from tabench.models import MODEL_REGISTRY

_REPO = Path(__file__).resolve().parents[1]
_TUT = _REPO / "tutorials"


def _track_manifest() -> dict[str, object]:
    """The 11 parallel-track units, each anchored to a public symbol so a rename or
    removal breaks THIS import rather than silently weakening the coverage gate."""
    import tabench.bottleneck as bn
    import tabench.dnl as dnl
    import tabench.dta as dta
    import tabench.newell as nw
    import tabench.tdta as td
    import tabench.transit as tr

    return {
        "transit-strategy": tr.optimal_strategy,
        "vickrey": bn.vickrey_worked_scenario,
        "vi-due": bn.due_closed_form,
        "merchant-nemhauser": dta.solve_so_dta,
        "lp-so-dta": dta.solve_cell_so_dta,
        "pm-td": td.solve_td_ue,
        "newell-3det": nw.newell_min,
        "ctm": dnl.CTMLink,
        "ltm": dnl.LTMLink,
        "godunov": dnl.GodunovLink,
        "node-model": dnl.TampereNode,
    }


# Same-ADR sibling folds: one notebook stem certifies several registry units.
_COVERS: dict[str, set[str]] = {
    "gls": {"gls", "prior"},  # ADR-002 static T2: the stale prior vs the GLS estimator
    "od-dynamic": {  # ADR-023 within-day dynamic estimation
        "od-dynamic-sim",
        "od-dynamic-seq",
        "prior-profile",
    },
}

# Units that register only under an optional extra (absent from a core-only env).
_GUARDED = {"implicit-ue-nn", "het-gnn", "sumo-marouter", "dtalite-tap", "spsa-sumo"}

# Canonical track -> folder number (PI directive 2026-07-16, ROADMAP phase pedagogy: the
# GitHub folder listing reads simple -> complex). A track folder MUST carry exactly this
# number; the numbering-integrity test binds it so ordering can never silently drift.
_CANON_TRACK_NUM = {
    "static": 1, "day-to-day": 2, "estimation": 3, "transit": 4, "dnl": 5,
    "bottleneck": 6, "dta": 7, "tdta": 8, "newell": 9, "learned": 10,
    "external": 11, "data": 12, "experiments": 13,
}

# THE shrinking allowlist — the single source of truth for not-yet-written units.
# INVARIANT: a batch may ONLY ever REMOVE entries from this set, never add. It shrinks
# monotonically as notebooks land and reaches empty in the final data/experiments batch,
# at which point the coverage gate is fully strict. `test_allowlist_is_honest` fails if
# an allowlisted unit already has a notebook (drift) or is not a real enforced unit
# (typo). Seeded for batch-00, where only static/bfw.ipynb exists.
_ALLOWLIST: set[str] = {
    # static (batches 01–05) + bfw have all shipped — the whole static track is done
    # day-to-day (batch-06) — SHIPPED, numbered 02-day-to-day/01..07
    # dnl (batch-07) — SHIPPED, numbered 05-dnl/01..04
    # analytic dta (batch-08) — SHIPPED (06-bottleneck/01..02, 07-dta/01..02,
    # 08-tdta/01)
    # newell + transit (batch-09) — SHIPPED, numbered 09-newell/01, 04-transit/01
    # estimation (batch-10) — SHIPPED, numbered 03-estimation/01..07 (prior +
    # the dynamic trio satisfied via _COVERS folds on 01-gls / 07-od-dynamic)
    # torch (batch-11) + engines (batch-12)
    "implicit-ue-nn", "het-gnn", "sumo-marouter", "dtalite-tap", "spsa-sumo",
}
# PIN: today's allowlist is exactly the 5 guarded torch/sumo/dtalite units — any re-add
# of a shipped unit must also bump this bound, forcing a second visible diff line beyond
# the set literal itself (the shrink-only invariant above is review-enforced, not
# mechanical; this narrows the blast radius of a silent re-add). Update the bound (and
# this comment) only when a NEW batch adds a genuinely not-yet-written guarded unit.
assert len(_ALLOWLIST) <= 5, "allowlist grew — update this pin's bound and comment"


def _enforced_units() -> list[str]:
    units = (
        list(MODEL_REGISTRY)
        + list(ESTIMATOR_REGISTRY)
        + list(DYNAMIC_ESTIMATOR_REGISTRY)
        + list(_track_manifest())
    )
    assert len(units) == len(set(units)), "unit-name collision across enforcement surfaces"
    return sorted(units)


# Numbered simple->complex layout (PI directive 2026-07-16): tutorials/<NN>-<track>/
# <MM>-<unit>.ipynb. The numeric prefix orders the GitHub listing as the learning path; the
# BARE name (unit / track) after it is the contract key, so every consumer strips the prefix.
_NUM_PREFIX = re.compile(r"^\d\d-(.+)$")


def _strip_num(name: str) -> str:
    """Drop a leading ``NN-`` ordering prefix from a folder or file stem, giving the bare
    track / unit name (the registry key). Names without a prefix are returned unchanged."""
    m = _NUM_PREFIX.match(name)
    return m.group(1) if m else name


def _existing_notebooks() -> list[Path]:
    # rglob (not a one-level glob) so nested planned-track dirs are found, but that also
    # walks Jupyter's own `.ipynb_checkpoints/` autosave dirs (created the moment a shipped
    # notebook is opened in Jupyter Lab, exactly what the README tells users to run) — filter
    # them out, or a contributor who merely opened a notebook gets spurious local failures.
    return sorted(
        p for p in _TUT.rglob("*.ipynb") if ".ipynb_checkpoints" not in p.parts
    )


def _notebook_units() -> set[str]:
    return {_strip_num(p.stem) for p in _existing_notebooks()}


def _unit_satisfied(unit: str, units: set[str]) -> bool:
    if unit in units:
        return True
    return any(unit in covered and stem in units for stem, covered in _COVERS.items())


@pytest.mark.parametrize("unit", _enforced_units())
def test_unit_has_tutorial_notebook(unit):
    if _unit_satisfied(unit, _notebook_units()):
        return
    assert unit in _ALLOWLIST, (
        f"registered unit '{unit}' has no tutorial notebook under "
        f"tutorials/<NN>-<track>/<MM>-{unit}.ipynb (and no same-ADR notebook covers it). "
        "Ship one, or — if it is an intentional not-yet-written unit — add it to the "
        "shrinking _ALLOWLIST in this file."
    )


def test_allowlist_is_honest():
    units = _notebook_units()
    # A unit that already has a notebook must not linger in the allowlist, or a later
    # regression (deleted notebook) would pass silently.
    stale = {u for u in _ALLOWLIST if _unit_satisfied(u, units)}
    assert not stale, f"allowlisted units already have notebooks — remove them: {sorted(stale)}"
    # Every allowlisted unit is a real enforced unit (guarded units may be absent from
    # a core-only env but are legitimately pre-listed for their extras batch).
    unknown = _ALLOWLIST - set(_enforced_units()) - _GUARDED
    assert not unknown, f"allowlist has entries that are not enforced units: {sorted(unknown)}"


def test_dnl_manifest_complete():
    """A new ``LinkModel`` / ``NodeModel`` subclass must join the manifest (walked
    transitively — GodunovLink subclasses CTMLink)."""
    import importlib
    import pkgutil

    import tabench.dnl as dnl

    # ``__subclasses__()`` only sees classes whose defining module has been imported, so
    # import EVERY dnl submodule first: the check must give the same verdict running this
    # file alone as in the full suite (where e.g. dnl tests import the private reference
    # link model). A new production link/node model in any module is then caught here.
    for mod in pkgutil.iter_modules(dnl.__path__):
        importlib.import_module(f"tabench.dnl.{mod.name}")

    def _tabench_subclasses(base) -> set[str]:
        # Two leakage paths into this walk: (1) the private reference link model, closed by
        # the pkgutil import above + the PointQueueLink waiver; (2) ad-hoc test-LOCAL
        # subclasses defined in OTHER test functions that linger in __subclasses__() until
        # gc — an intermittent full-suite failure. The module filter closes (2): only
        # first-party ``tabench.`` classes are real units; foreign scratch classes are
        # excluded regardless of gc timing.
        found: set[str] = set()
        for cls in base.__subclasses__():
            if cls.__module__.startswith("tabench."):
                found.add(cls.__name__)
            found |= _tabench_subclasses(cls)
        return found

    covered = {
        "CTMLink", "LTMLink", "GodunovLink", "TampereNode",  # the four manifest units
        "SeriesNode", "OriginNode", "DestinationNode",  # waived: boundary plumbing
        "PointQueueLink",  # waived: private reference impl (tabench.dnl._reference), no notebook
    }
    missing = (_tabench_subclasses(dnl.LinkModel) | _tabench_subclasses(dnl.NodeModel)) - covered
    assert not missing, f"DNL classes missing from the tutorial manifest: {sorted(missing)}"

    # PIN (m12): a scratch subclass defined HERE (module != 'tabench.*') and NOT gc'd must
    # not trip the gate — the module filter must exclude it deterministically.
    class _ScratchLink(dnl.LinkModel):
        pass

    class _ScratchNode(dnl.NodeModel):
        pass

    leaked = (_tabench_subclasses(dnl.LinkModel) | _tabench_subclasses(dnl.NodeModel)) - covered
    assert not leaked, f"foreign scratch subclasses leaked into the gate: {sorted(leaked)}"


@pytest.mark.parametrize(
    "nb_path", _existing_notebooks(), ids=lambda p: str(p.relative_to(_TUT))
)
def test_notebook_is_stripped(nb_path):
    nb = json.loads(nb_path.read_text())
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        assert cell.get("outputs", []) == [], f"{nb_path.name}: committed code outputs (strip)"
        assert cell.get("execution_count") is None, f"{nb_path.name}: execution_count not cleared"


@pytest.mark.parametrize(
    "nb_path", _existing_notebooks(), ids=lambda p: str(p.relative_to(_TUT))
)
def test_notebook_metadata_consistent(nb_path):
    tab = json.loads(nb_path.read_text()).get("metadata", {}).get("tabench")
    name = nb_path.name
    assert tab is not None, f"{name}: missing metadata.tabench block"
    # metadata carries the BARE names (registry keys); the on-disk names add an NN- ordering
    # prefix, so strip it before comparing (metadata.unit is stable across a renumber).
    assert tab.get("unit") == _strip_num(nb_path.stem), (
        f"{name}: metadata.tabench.unit != filename stem (sans NN- prefix)"
    )
    assert tab.get("track") == _strip_num(nb_path.parent.name), (
        f"{name}: metadata.tabench.track != folder name (sans NN- prefix)"
    )
    assert tab.get("requires_extra") in (None, "torch", "sumo", "dtalite"), (
        f"{name}: bad metadata.tabench.requires_extra {tab.get('requires_extra')!r}"
    )
    # metadata.tabench.covers must actually BE the enforcement source (README says the
    # gate "keys off metadata.tabench, not off filenames"): pin it equal to the module's
    # own _COVERS entry for this stem (itself excluded — a notebook doesn't "cover" its
    # own unit), empty when the stem folds no siblings. A future drift between the two
    # would otherwise be silently accepted (only _COVERS is actually read by the fold
    # logic); this makes metadata.covers load-bearing, not decorative.
    stem = _strip_num(nb_path.stem)
    expected_covers = _COVERS.get(stem, set()) - {stem}
    assert set(tab.get("covers") or []) == expected_covers, (
        f"{name}: metadata.tabench.covers {tab.get('covers')!r} != "
        f"_COVERS[{stem!r}] {sorted(expected_covers)!r}"
    )


def test_notebook_numbering_is_ordered():
    """The numbered simple->complex layout cannot silently drift or collide. Each track
    folder is ``<NN>-<track>`` with NN the CANONICAL track number and unique across folders;
    within a folder the files are ``<MM>-<unit>.ipynb`` with MM unique and contiguous from 01.

    Cross-folder *contiguity* is a final-state property only and is deliberately NOT enforced:
    tracks ship out of numeric order across batches (02-day-to-day lands before 03-estimation),
    so requiring 01..N with no gaps mid-rollout would red the suite on a correct partial tree.
    Uniqueness + canonical-number match is the invariant that actually prevents collisions."""
    seen: dict[int, str] = {}
    for d in sorted(p for p in _TUT.iterdir() if p.is_dir()):
        m = _NUM_PREFIX.match(d.name)  # reuses ^\d\d-(.+)$
        assert m, f"track folder {d.name!r} lacks an NN- numeric prefix"
        num, track = int(d.name[:2]), m.group(1)
        assert track in _CANON_TRACK_NUM, f"unknown track folder {track!r}"
        assert num == _CANON_TRACK_NUM[track], (
            f"folder {d.name!r}: track {track!r} must be numbered "
            f"{_CANON_TRACK_NUM[track]:02d}, not {num:02d}"
        )
        assert num not in seen, f"duplicate track number {num:02d}: {seen[num]!r} and {track!r}"
        seen[num] = track
        nums = []
        for p in d.glob("*.ipynb"):
            fm = _NUM_PREFIX.match(p.stem)
            assert fm, f"notebook {p.name!r} in {d.name!r} lacks an NN- numeric prefix"
            nums.append(int(p.stem[:2]))
        nums.sort()
        assert nums == list(range(1, len(nums) + 1)), (
            f"{d.name!r} file numbers must be unique and contiguous from 01, got {nums}"
        )


# requires_extra -> the module name to probe. `import DTALite` is BANNED in-process
# (adr-029: it prints a banner and ctypes-loads the engine .so into the host), and the
# extra name 'dtalite' is not even the module name ('DTALite' is) — so probe EVERY extra
# with find_spec, which never imports. Uniform and side-effect free.
_EXTRA_MODULE = {"torch": "torch", "sumo": "sumo", "dtalite": "DTALite"}


def _extra_available(extra: str) -> bool:
    return importlib.util.find_spec(_EXTRA_MODULE[extra]) is not None


def test_extra_gate_probes_dtalite_by_module_name_via_find_spec():
    # Regression pin: dtalite must be probed as the module 'DTALite' with find_spec.
    # `importorskip('dtalite')` would skip FOREVER (wrong name), and `import DTALite`
    # is banned in-process (adr-029). find_spec never imports.
    assert _EXTRA_MODULE["dtalite"] == "DTALite"
    assert importlib.util.find_spec("dtalite") is None  # the extra name is not a module
    assert _extra_available("dtalite") == (importlib.util.find_spec("DTALite") is not None)


@pytest.mark.skipif(
    not os.environ.get("TABENCH_RUN_TUTORIALS"),
    reason="set TABENCH_RUN_TUTORIALS=1 to execute notebooks (CI does; laptops skip)",
)
@pytest.mark.parametrize(
    "nb_path", _existing_notebooks(), ids=lambda p: str(p.relative_to(_TUT))
)
def test_notebook_executes(nb_path):
    # Discovery is filesystem + metadata driven (rglob above, requires_extra below):
    # no hardcoded notebook list to fall out of sync with what actually ships.
    nbformat = pytest.importorskip("nbformat")
    nbclient = pytest.importorskip("nbclient")
    nb = nbformat.read(nb_path, as_version=4)
    extra = nb.get("metadata", {}).get("tabench", {}).get("requires_extra")
    if extra and not _extra_available(extra):
        pytest.skip(f"optional extra {extra!r} not installed")
    nbclient.NotebookClient(nb, timeout=120, kernel_name="python3").execute()
