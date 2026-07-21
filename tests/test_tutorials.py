"""Enforcement for the per-unit tutorial notebooks (``tutorials/<NN>-<track>/<MM>-<unit>.ipynb``).

The PI rule "a new model must ship a tutorial" made mechanical:

* every key of ``MODEL_REGISTRY`` / ``ESTIMATOR_REGISTRY`` /
  ``DYNAMIC_ESTIMATOR_REGISTRY`` and every parallel-track unit (an import-anchored
  11-unit manifest — those tracks carry no registry) maps to a notebook, allowing
  same-ADR ``covers`` folds;
* the DNL track cannot grow silently — new ``LinkModel`` / ``NodeModel`` subclasses
  must appear in the manifest;
* notebooks are committed EXECUTED — every non-empty code cell carries the outputs
  and the strictly sequential ``execution_count`` of one clean top-to-bottom run
  (PI directive 2026-07-21: the committed notebook IS the rendered tutorial).
  Drift safety still comes from the CI re-execution below, never from output
  identity. The one exemption ships stripped: the matsim notebook, whose
  JAVA-only toolchain (adr-039) is absent on the maintainer box; the matsim CI
  job still proves its executability. ``metadata.tabench`` stays
  folder-consistent;
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

    manifest: dict[str, object] = {
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
    # EDOC producers are not in MODEL_REGISTRY, so they bind to the coverage gate here.
    # Guarded like the adapter's own import (sumo is optional): enforced on the sumo leg,
    # invisible on core-only legs (no collection break, no false enforcement).
    try:
        from tabench.models.adapters.sumo_duaiterate import SumoDuaIterateAdapter
        manifest["sumo-duaiterate"] = SumoDuaIterateAdapter
    except ModuleNotFoundError as exc:
        if exc.name != "sumo":
            raise
    # matsim (adr-039) imports EVERYWHERE (Java-only engine, no python extra), so
    # its entry is UNCONDITIONAL — the coverage gate then enforces the notebook on
    # ALL legs, which the same-commit notebook satisfies (the S2 atomicity
    # pattern; _ALLOWLIST stays empty).
    from tabench.models.adapters.matsim_edoc import MatsimAdapter
    manifest["matsim"] = MatsimAdapter
    # dtalite-simulation (adr-040) likewise imports EVERYWHERE — the module never
    # imports the DTALite wheel in-host (subprocess-only engine; find_spec probe),
    # so its entry is UNCONDITIONAL and the coverage gate enforces the notebook on
    # all legs, satisfied atomically by the same-commit notebook (_ALLOWLIST stays
    # empty). NOT in MODEL_REGISTRY (EDOC producer).
    from tabench.models.adapters.dtalite_simulation import DTALiteSimulationAdapter
    manifest["dtalite-simulation"] = DTALiteSimulationAdapter
    # bo4mob-estimation (adr-041): a NEW T2 sibling family (BO4MOB_ESTIMATOR_REGISTRY,
    # not MODEL/ESTIMATOR/DYNAMIC), so it binds to the coverage gate HERE. Its certifier
    # ALWAYS runs od2trips+meso, so the notebook is requires_extra=sumo (a hard gate, the
    # spsa-sumo precedent). bo4mob_base imports WITHOUT sumo (the prior baseline registers
    # unconditionally), so guard the entry on a sumo find_spec — enforced on the sumo leg,
    # invisible on core-only. The notebook still ships in the tree (same-commit atomic
    # pattern; _ALLOWLIST stays empty).
    if importlib.util.find_spec("sumo") is not None:
        from tabench.estimation.bo4mob_base import Bo4MobPriorBaseline
        manifest["bo4mob-estimation"] = Bo4MobPriorBaseline
    return manifest


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
# static (batches 01–05) + bfw have all shipped — the whole static track is done
# day-to-day (batch-06) — SHIPPED, numbered 02-day-to-day/01..07
# dnl (batch-07) — SHIPPED, numbered 05-dnl/01..04
# analytic dta (batch-08) — SHIPPED (06-bottleneck/01..02, 07-dta/01..02, 08-tdta/01)
# newell + transit (batch-09) — SHIPPED, numbered 09-newell/01, 04-transit/01
# estimation (batch-10) — SHIPPED, numbered 03-estimation/01..07 (prior + the dynamic
# trio satisfied via _COVERS folds on 01-gls / 07-od-dynamic)
# torch (batch-11) + engines (batch-12) — all SHIPPED:
# sumo-marouter — 11-external/01-sumo-marouter.ipynb
# implicit-ue-nn — 10-learned/01-implicit-ue-nn.ipynb
# het-gnn — 10-learned/02-het-gnn.ipynb
# dtalite-tap — 11-external/03-dtalite-tap.ipynb
# spsa-sumo — 11-external/04-spsa-sumo.ipynb
#
# `{}` is a DICT literal in Python, not an empty set — set() is required so this stays
# the set _unit_satisfied/_ALLOWLIST arithmetic below expects.
_ALLOWLIST: set[str] = set()
# PIN: EMPTY (the C3 closing pass, S0b) — every enforced unit now has a shipped notebook,
# so the coverage gate is fully strict: any future model/estimator/parallel-track unit
# added without a tutorial fails test_unit_has_tutorial_notebook immediately, with no
# allowlist escape hatch left to re-open. Re-adding an entry here is therefore a real
# regression, not routine batch bookkeeping — it must name why coverage is going
# backwards, not just bump this bound.
assert len(_ALLOWLIST) <= 0, "allowlist grew — update this pin's bound and comment"


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


# The notebooks that CANNOT be executed on the maintainer box and therefore ship
# stripped (no outputs, no execution counts). matsim is JAVA-only behind no pip extra
# (adr-039: TABENCH_MATSIM_HOME + TABENCH_JAVA_HOME); its executability is still proven
# by the matsim CI job, which executes it from a cleared state. Shrink, never grow.
_EXECUTED_EXEMPT = frozenset({"11-external/05-matsim.ipynb"})


@pytest.mark.parametrize(
    "nb_path", _existing_notebooks(), ids=lambda p: str(p.relative_to(_TUT))
)
def test_notebook_is_executed(nb_path):
    nb = json.loads(nb_path.read_text())
    code = [c for c in nb.get("cells", []) if c.get("cell_type") == "code"]
    if str(nb_path.relative_to(_TUT)) in _EXECUTED_EXEMPT:
        for cell in code:
            assert cell.get("outputs", []) == [], (
                f"{nb_path.name}: exempt notebooks ship stripped (committed outputs)"
            )
            assert cell.get("execution_count") is None, (
                f"{nb_path.name}: exempt notebooks ship stripped (execution_count set)"
            )
        return
    # One clean top-to-bottom run: non-empty code cells count strictly 1..N. Empty
    # cells are never executed by nbclient and stay None by construction.
    nonempty = [c for c in code if "".join(c.get("source", "")).strip()]
    counts = [c.get("execution_count") for c in nonempty]
    assert counts == list(range(1, len(nonempty) + 1)), (
        f"{nb_path.name}: not one clean top-to-bottom execution "
        f"(execution counts {counts}); re-run the whole notebook, do not commit "
        "a partially or out-of-order executed state"
    )


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
    assert tab.get("requires_extra") in (None, "torch", "sumo", "dtalite", "matsim"), (
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


# requires_extra -> the probe. torch/sumo/dtalite probe by MODULE NAME via find_spec
# (BYTE-FOR-BYTE the pre-S3 behavior: `import DTALite` is BANNED in-process — adr-029:
# it prints a banner and ctypes-loads the engine .so into the host — and the extra name
# 'dtalite' is not even the module name; find_spec never imports). 'matsim' has NO
# python module at all (Java-only engine, adr-039; PyPI 'matsim' is an unrelated
# neuronal simulator), so its probe is the adapter's side-effect-free runtime
# availability CALLABLE (env-var + jar + java path checks; no JVM is started).
from tabench.models.adapters._matsim_io import matsim_available  # noqa: E402

_EXTRA_MODULE = {
    "torch": "torch",
    "sumo": "sumo",
    "dtalite": "DTALite",
    "matsim": matsim_available,
}


def _extra_available(extra: str) -> bool:
    probe = _EXTRA_MODULE[extra]
    if callable(probe):
        return bool(probe())
    return importlib.util.find_spec(probe) is not None


def test_extra_gate_probes_dtalite_by_module_name_via_find_spec():
    # Regression pin: dtalite must be probed as the module 'DTALite' with find_spec.
    # `importorskip('dtalite')` would skip FOREVER (wrong name), and `import DTALite`
    # is banned in-process (adr-029). find_spec never imports.
    assert _EXTRA_MODULE["dtalite"] == "DTALite"
    assert importlib.util.find_spec("dtalite") is None  # the extra name is not a module
    assert _extra_available("dtalite") == (importlib.util.find_spec("DTALite") is not None)


def test_extra_gate_probes_matsim_by_callable():
    # Regression pin (adr-039): matsim must be probed via the adapter's runtime
    # callable — a find_spec probe would either skip forever (no such module) or,
    # worse, match the unrelated PyPI 'matsim' neuronal simulator (adr-030).
    assert _EXTRA_MODULE["matsim"] is matsim_available
    assert _extra_available("matsim") == matsim_available()


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
    # Per-CELL cap. 300 s (was 120): the matsim notebook's negative-control cell
    # runs 2 states x 5 macroreps x (one JVM co-evolution + 2 JVM replays) in ONE
    # cell — measured ~112 s on the dev box (adr-039), so 120 s would flake on a
    # slower runner. Still a hang-stop, not a budget (the docs build allows 600).
    nbclient.NotebookClient(nb, timeout=300, kernel_name="python3").execute()
