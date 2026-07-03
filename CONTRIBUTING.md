# Contributing to TABenchmark

TABenchmark exists to make *every* notable traffic assignment model testable on shared
scenarios. Contributions of models, networks, observation levels, metrics, and reference
implementations are all welcome. Read [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
first — the design principles (P1–P9) are normative for review.

## Contributing a model

1. **Implement the contract.** Subclass `tabench.models.TrafficAssignmentModel`
   (or wrap your artifact with `CallableModel` / a subprocess adapter):
   - declare `capabilities` honestly (paradigm, determinism, `provides_gap`,
     `seedable`, `trained_on` lineage for learned models);
   - declare tunable hyperparameters as `factors` with defaults;
   - implement `solve(scenario, budget, rng, trace)`, record at least one checkpoint,
     and respect the budget.
2. **Do not compute your own score.** Self-reported gaps go into checkpoint
   `self_report` entries; the harness recomputes everything (P1). White-box solvers
   should self-report anyway — the harness diffs the two as an honesty check.
3. **Randomness** must come only from `rng.generator(source=i)` streams (P8). If your
   model wraps an unseedable engine, declare `seedable=False`.
4. **Learned models** must declare their complete training lineage in
   `capabilities.trained_on` (scenario families or content hashes). Evaluations that
   intersect the lineage are refused; undeclared contamination discovered later means
   removal of results.

### Conformance checklist (reviewed on every model PR)

- [ ] Solves the built-in Braess scenario; certified gap consistent with the model
      class (equilibrium solvers: analytic UE reached within documented budget)
- [ ] Runs twice with the same seed → identical trace (if `seedable=True`)
- [ ] Budget respected on every declared axis
- [ ] Declared outputs actually emitted; flows pass the feasibility audit
- [ ] `pytest` and `ruff check` clean; a test file exercising the model exists
- [ ] Reference(s) for the method exist in `docs/REFERENCES.md` (add them with
      verified BibTeX if missing — no unverified citations)

## Contributing a network / scenario

- Never commit network data (P9). Add a `NetworkSpec` to `tabench/data/registry.py`
  with commit-pinned SHA-256 checksums, per-network **units metadata**, known defects,
  and the mandatory citation.
- Add a declarative YAML entry under `scenarios/` at the appropriate ladder rung.
- If a best-known solution exists, wire it as a `ReferenceSolution` and add a
  regression test that recomputes the oracle objective from the flows (never hardcode
  unit-dependent constants).

## Contributing an observation level

Subclass `tabench.observe.DataLevel`. The process must be a pure function of
`(ground truth, rng)`, document its dials in `meta`, and ship a determinism test.
Distribute per-period data, never time-averages (Hazelton 2015 — the dependence
pattern carries information).

## Code style

Python ≥ 3.10, type hints, docstrings, PEP 8 via `ruff` (line length 100). Keep core
dependencies to NumPy/SciPy/PyYAML; heavier dependencies (torch, engines) belong in
optional extras or adapters.

## Scope

v0 covers static assignment on TNTP networks. The roadmap
([docs/ROADMAP.md](docs/ROADMAP.md)) stages SUE, bush-based solvers, estimation tasks,
DTA, and engine adapters — PRs that advance a roadmap tier are especially welcome.
