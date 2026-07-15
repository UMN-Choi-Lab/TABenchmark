# ADR-029 — DTALite `assignment()`: the second external engine, and the identity-map static-UE row

**Status:** accepted (shipped in v0.2)
**File:** `docs/design/adr-029-dtalite-tap.md`

## Context — the second external simulator, and a cost law that maps EXACTLY

ADR-027 shipped `sumo-marouter`, the first external simulator: its hardcoded
linear-in-flow class law forced a *mapping floor* (Braess RG ~1.7e-4) because the
scenario's BPR could only be approximated. This ADR is the second external engine,
the named `dtalite-adapter` sprint (canon `zhou2014dtalite`, tier 1,
`docs/REFERENCES.md`): the PyPI **`DTALite`** wheel (Fang Tang + Xuesong Zhou,
0.8.1, ~1.3 MB, no declared dependencies, ctypes-loads a compiled engine). Its
`assignment()` entry is a **static Frank-Wolfe user-equilibrium solver** with a
**per-link BPR VDF** — so unlike marouter the compile map is the **identity** on
`(free_flow_time, b, power, capacity)`, and Sioux Falls power-4 becomes the marquee
anchor rather than an "unrepresentable" refusal.

Everything below was **verified by executing the pip wheel** in this repo's python
(`pip install DTALite==0.8.1`) against the repo's own scenarios and certifying the
emitted flows through the UNCHANGED harness (P1). Three engine hazards make the
subprocess wrapper mandatory rather than merely hygienic (§4).

## Decision 1 — dependency mechanics (the adr-025/027 optional-extra precedent)

- **Extra:** `dtalite = ["DTALite>=0.8"]` in `[project.optional-dependencies]`, a
  **floor not a pin** (a `==` in library metadata conflicts with user environments).
  The FW/VDF behavior lives in a bundled `.so` that can shift under the floor, so the
  CI workflow pins `DTALite==0.8.1` (the tested engine) as the anchor tripwire.
- **Guard — one deliberate deviation from the sumo shape.** `import DTALite` has two
  side effects the sumo guard never had to consider: it **prints a version banner to
  stdout** and **ctypes-loads the engine `.so` + libgomp into the host process**. So
  the adapter module NEVER imports `DTALite` at module scope; it probes availability
  with `importlib.util.find_spec("DTALite")` and raises
  `ModuleNotFoundError(name="DTALite")` when absent. `models/adapters/__init__.py` and
  `models/__init__.py` swallow that by **exact name** (`exc.name != "DTALite"` re-raises
  — note the case: the module is `DTALite`, not `dtalite`). Engine-version provenance
  comes from `importlib.metadata.version("DTALite")`, never an import. A regression
  asserts `import tabench` prints **nothing** to stdout when the model registers (the
  banner-suppression test unique to this adapter), and that blocking `DTALite` via a
  meta_path finder unregisters the model while the numpy core still imports.
- **CI: a FIFTH job `dtalite`**, a clone of the sumo/torch job shape (py3.12, checkout,
  setup-python, data cache + prefetch + `TABENCH_REQUIRE_DATA=1`, `pip install
  DTALite==0.8.1` before `pip install -e ".[dev,dtalite]"`, then only
  `pytest tests/test_dtalite.py`). **No pip-cache step** (the 1.3 MB wheel does not
  warrant it — the cache overhead exceeds the download). Kept separate from the sumo
  job (not folded) so either dependency's breakage cannot kill the other's signal
  (the ADR-027 Decision-1 reasoning; here the cost asymmetry that might justify folding
  does not exist — the wheel is tiny). actionlint-clean.

## Decision 2 — the model contract

`DTALiteTapModel` (`dtalite-tap`), **registered** with class-level capabilities, in
`src/tabench/models/adapters/dtalite_tap.py` (one module — the mapping is a near-identity
so no separate `_io` module clarifies anything).

- **Paradigm: reuse `"heuristic"`** (the `aon`/`sumo-marouter` precedent). The Evaluator
  branches on the scenario's task fields, not the paradigm, so a heuristic-paradigm model
  earns the certified `relative_gap`/AEC/Beckmann on static scenarios exactly like `aon`.
  We do NOT declare `static_ue`: white-box status means "internals match the scenario's
  declared costs", and while the engine IS a genuine FW UE solver on the exactly-mapped
  BPR, the certified gap tells the truth either way. **No `capabilities.py` change, no
  Evaluator branch, no scenario field, no hash change.**
- **`deterministic=True, provides_gap=False, seedable=False`.** The FW loop has no RNG:
  `link_performance.csv` is byte-identical across reruns at `number_of_processors=1`
  (re-confirmed in-tree by md5). Unlike marouter there is no command line to pin a
  provenance seed on, so `seedable=False` (the RngBundle root seed still lands in the
  manifest via `seed_info`). `provides_gap=False`: the harness recomputes the gap; the
  engine's self-reported gap is recorded as provenance only (§ Measured, it uses a
  different normalization).
- **Budget (P6).** `iterations → number_of_iterations` (floored at **1**); `wall_seconds`
  is threaded as a **single deadline through all three phases** (write inputs → engine
  subprocess → parse); `sp_calls` is unmappable (the engine hides its Dijkstra count):
  `coords.sp_calls = 0`, and an **sp_calls-ONLY budget raises `ValueError`** up front —
  never silently unbounded. `target_relative_gap` is ignored (the engine runs a fixed
  count, no gap-target hook) — disclosed. Unlike ADR-027, the executed iteration count
  IS recoverable from `summary_log_file.txt`, so `coords.iterations` reports the
  **executed** FW count and `self_report` carries the configured count.
- **Error surfaces, three distinct.** (a) missing `DTALite` → absent from registry
  (guard); (b) nonzero exit, timeout, `.so` load failure in the child, missing/empty
  `link_performance.csv`, or a read-back mismatch → **`RuntimeError`** with command +
  stdout/stderr tails (an engine failure is infrastructure, NEVER laundered into
  `feasible=0`); (c) emitted-but-bad flows → the normal harness censor path, zero adapter
  logic. Zero off-diagonal demand → in-adapter exact-zero short-circuit before any
  subprocess.
- **Hygiene.** Each solve runs in a `tempfile` dir prefixed `tabench-dtalite-` (cleanup
  guaranteed on raise via `finally`); the `keep_files` factor keeps it (`last_workdir`).
  The child's CWD confinement means the engine's ≥9 output CSVs never touch the repo/CWD.

## Decision 3 — the compile map is the IDENTITY (the central content)

The engine's per-link VDF, measured and source-confirmed, is
`t = vdf_fftt · (1 + vdf_alpha · (I/Link_Capacity)^vdf_beta)` with
`I = V/lanes/period_hours/vdf_plf` and `Link_Capacity = lanes·capacity`. Writing
**`lanes = 1`**, a **1-hour demand period** (07:00–08:00), **`vdf_plf = 1`**, and the
repo values verbatim (`vdf_fftt = free_flow_time`, `vdf_alpha = b`, `vdf_beta = power`,
`capacity = capacity`, `toll = 0`) collapses this to the repo BPR
`t = fft·(1 + b·(v/cap)^power)` **exactly**. Five engine traps are neutralized:

- **Grouped-input requirement (the review CRITICAL).** The engine builds its adjacency
  from contiguous `FirstLinkFrom/LastLinkFrom` node ranges, so `link.csv` **must** be
  written grouped/sorted by `(from_node_id, to_node_id)`. An ungrouped file silently
  corrupts routing: a permuted Braess certifies `feasible=1` at the WRONG flows (RG 0.208
  vs the true 0.0118) and a permuted Sioux Falls sends the FW loop into an **infinite
  loop** (which, under an iterations-only budget with no wall, hangs `solve()` forever).
  The adapter now writes links sorted by node pair; the read-back matches by node pair, so
  the emitted order is transparent to it and the solve is permutation-invariant.
- **The capacity `fmax(0.1, cap)` clamp (the review MAJOR).** The engine clamps
  `capacity` at `fmax(0.1, cap)` **in the cost law only** (`Link_Travel_Time`; the Beckmann
  integral is left unclamped — internally inconsistent below 0.1), so a capacity in
  `(1e-4, 0.1)` equilibrates under a DIFFERENT BPR while passing the echo read-back
  (measured A2 rel 0.93, certified `feasible=1` RG 0.799 with false "pure FW truncation"
  provenance). The adapter refuses any `capacity < 0.1`, naming the clamp. The engine's
  other `fmax` floors — lanes `0.01`, period `0.001`, plf `0.0001` — are all neutralized by
  our fixed `lanes=1 / period=1 h / plf=1`.
- **The `lanes²` trap** (measured): for `lanes = L` the engine divides the ratio by
  `L²·capacity` while its own `doc` column divides by `L·capacity` — the two agree only
  at `L = 1`. The adapter ALWAYS writes `lanes = 1` with `capacity` = the total link
  capacity, so the assignment-driving cost is the textbook BPR.
- **Case sensitivity** (measured): the engine matches column names case-sensitively and
  silently ignores an uppercased column (`VDF_Alpha` → default 0.15). The adapter writes
  lowercase columns and the read-back (below) catches any that were ignored.
- **`vdf_fftt` override**: writing `vdf_fftt` explicitly overrides the
  `length/free_speed`-derived free-flow time, so the abstract native cost units pass
  through verbatim (no mph/km conversion enters the cost). Confirmed: Sioux Falls
  `fft = 6.0` echoes `6.0000` exactly.

**Read-back (the house compile-check) — echo checks plus a LIVE A2 cost-match.**
`link_performance.csv` echoes `vdf_fftt/vdf_alpha/vdf_beta/vdf_plf/link_capacity` per
link. The adapter verifies every repo link is matched exactly once by `(from_node_id,
to_node_id)`, the row count equals the link count (no phantom/dropped link), each echoed
parameter equals the declared value within the engine's **float32/4-decimal** precision,
and the volume is finite. Because the echo check is **vacuous below its `atol = 1e-3`
absolute floor** (a doctored/ignored sub-1e-3 `vdf_fftt` is accepted — the review MINOR),
the LIVE gate is a **runtime A2 cost-match**: the engine's own `travel_time` column must
equal `network.link_cost(flows)` on EVERY link (the identity map's payoff), to relative
`1e-3` (measured legit max ~2e-5 on the anchors, ~7.6e-4 on Barcelona's power-16.83 links).
This closes the echo blind spot AND catches the capacity clamp (rel 0.93) and an ignored
`vdf_fftt` (rel 1). Every cell conversion and the file read are wrapped so a
corrupt-but-parseable output raises the contract `RuntimeError` (with stdout+stderr tails),
never a raw `ValueError`/`TypeError` that would collide with the `ValueError` refusal
channel (the review MAJOR). A mismatch raises `RuntimeError` — this is what makes
`returncode == 0` untrustworthy safe: the engine exits 0 on missing files, dropped links,
and `zone_id != node_id` with zero/garbage flows, so success is DEFINED as the read-back
(echo + A2 + mass) matching, not the exit code.

**BIG-M ceiling → per-origin mass gate (the review MAJOR).** The engine's ~`1e7`
generalized-cost ceiling silently ZEROES an OD whose congested cost exceeds it (measured:
a single link `cap=100, power=4` carries `demand=9000` but emits `0.0` at `demand=10000`),
producing a well-formed all-zero-flow row that passes the echo/A2 read-back and would
otherwise launder to `feasible=0`. So after the read-back the adapter runs a **per-origin
mass gate**: each origin zone's emitted outflow must cover its routable demand (through
traffic only ADDS, so a shortfall is always an engine drop, never a solution — measured
deficit on clean inputs is 0, far below the 4-decimal rounding floor ~1e-4). A deficit
raises `RuntimeError` with the engine diagnostics. The per-origin (not merely total) check
covers the multi-OD partial-drop variant.

**Wall deadline through the parse phase (the review MINOR).** The single `wall_seconds`
deadline is checked after the subprocess AND after the read-back, so a slow large-network
parse cannot silently overrun the budget the ADR promises to enforce end-to-end.

**Zone convention.** Zones are nodes `1..n_zones` with `zone_id == node_id` (the hard
engine check); other nodes get `zone_id = 0`. `settings.csv first_through_node_id` carries
the scenario's TNTP `first_thru_node` verbatim — verified to route Sioux Falls (ftn = 1,
all 24 nodes are zones) correctly, all 76 links matched.

## Measured anchors (all certified by the UNCHANGED harness under the true BPR, P1)

Numbers are what the 0.8.1 wheel produced in-tree; bounds in the tests are loose ceilings
calibrated to them (version-robust — the `.so` can shift under the floor).

- **A1 — Braess (builtin, power=1, the fft=1e-6/alpha=1e7 sentinels).** `feasible = 1`,
  node balance 0, certified **RG 1.183e-2** at 100 iterations, **frozen from ~iter 40**
  (the engine's Armijo line search collapses to step 0 — the honest convergence ceiling,
  §"What it means"). The golden Braess content hash `cf00f411…` is re-asserted
  byte-identical. **The sentinels survive the parse**: `vdf_fftt = 1e-6` echoes as `0.0`
  (4-decimal display rounding) but the internal cost still matches (A2 below);
  `vdf_alpha = 1e7` echoes exactly. The representation survives the engine's CSV float
  parse — no silent corruption, so the row ships.
- **A2 — cost-matched anchor (MANDATORY).** The engine's own `travel_time` column equals
  `scenario.network.link_cost(v)` at the emitted `volume` on **every** link (the identity
  map's payoff — there is no mapping floor to separate out, unlike marouter): measured max
  **relative** error **7.1e-6 on Braess** (the two sentinel links; the clean links match
  to ~1e-15) and **2.2e-5 on Sioux Falls power-4** (dominated by the engine's 4-decimal
  `travel_time` column, not a cost-model error). The test pins `< 1e-3`. This proves the
  certified gap is pure FW truncation, not a cost mismatch — the row does not ship without
  it.
- **A3 — two-route deterministic UE.** `two_route_scenario(sue_theta=None)`: `feasible = 1`,
  **RG = 0.0** (exact UE, flows `(2.5, 2.5, 1.5, 1.5)`). Needs **no theta calibration at
  all** (contrast marouter's calibrated logit theta) — the engine solves deterministic UE
  natively, and on this well-conditioned instance exactly.
- **A4 — Sioux Falls power-4 (the marquee, marouter-impossible).** `feasible = 1`, all 76
  links matched, certified **RG 5.034e-3** at 100 iterations (frozen from ~iter 100), flow
  **NRMSE 1.58%** vs the best-known TNTP UE flows, A2 max-rel 2.2e-5. **The first external
  engine on the power-4 ladder** — where marouter's linear law refused, DTALite's per-link
  `vdf_beta` maps it exactly. The `~5.0e-3` stall floor is a Braess/Sioux-Falls figure;
  larger nets converge tighter (below).
- **Big-net exposure (the review's first large-network run).** At the scenario budget
  (iterations=500), both mixed-power TNTP nets certify with the sort fix and pass the A2 /
  mass gates: **Winnipeg** (2836 links, powers 0–6.87) `feasible=1`, RG `1.176e-3`, A2 max
  rel `5.2e-4`, wall ~7 s; **Barcelona** (2522 links, powers up to 16.83) `feasible=1`, RG
  `5.48e-4`, A2 max rel `7.6e-4` (the tightest legit A2 margin under the `1e-3` gate), wall
  ~5.2 s. On both, the engine self-gap equals `RG/(1−RG)` to 6 digits (the converged
  regime, below).
- **Negative controls + review regressions.** `iterations=1` runs the pure all-or-nothing
  load (0 FW line searches — measured: `number_of_iterations = 1` executes zero loop
  iterations after the initial AON), certifying `feasible = 1` with a much worse gap
  (Braess RG 1.9e-1) — an honest near-AON row that also pins the dial live: **RG(1) >
  RG(100)** (compared at iteration 1, not an intermediate — the gap freezes from ~iter 40,
  so `5 vs 100` is a plateau). Byte-determinism rerun. The refusal matrix (below). A tiny
  `wall_seconds` RAISES `RuntimeError`, and a slow parse RAISES too (the deadline covers
  the parse phase). Permuted link arrays reproduce the unpermuted solve (the sort fix).
  The capacity clamp refuses; the BIG-M drop raises; garbage engine cells and a wrong
  `travel_time` raise `_ReadBackError` (the A2 gate); a torn summary pair keeps the last
  complete `(iter, gap)`. Read-back rejects a short/phantom output and an ignored VDF
  parameter. Temp-dir hygiene on the normal and raising paths.

**The iterations off-by-one, resolved by measurement.** `number_of_iterations = N` logs
Frank-Wolfe iterations `1..N-1` in `summary_log_file.txt` (N−1 line searches after an
initial AON) and labels the `link_performance.csv` row `iteration_no = N`. So DOSSIER B's
source read (N → N−1) is correct, not DOSSIER A's "100 → 100". `N = 1` → 0 line searches =
the pure AON (valid, non-empty); `N = 0` → accessibility mode (empty output). So **the
floor is 1** (the AON is a legitimate near-AON row); `coords.iterations` reports the
executed N−1, and `self_report.engine_configured_iterations = N`.

**The self-reported gap definition, resolved by measurement.** The engine's printed gap is
`(TSTT − SPTT)/max(0.1, SPTT)` in percent (a clamped SPTT/least-TT denominator), whereas
the repo's certified `relative_gap` normalizes by TSTT. **On converged anchors** it equals
`RG/(1−RG)` to 6 digits (verified on Sioux Falls, Winnipeg and Barcelona; Braess engine
0.011978 vs `RG/(1−RG)` 0.011976), so it is **NOT** the repo gap. Off the converged regime
it can be **negative or frozen** (a stalled instance emits a stale/`SPTT>TSTT` gap; the
0.1 denominator clamp distorts it further). It is recorded as
`self_report.engine_relative_gap` (provenance only); the row is certified on the harness
value alone, and the self-report is **never gated on**.

## Refusals (naming the field)

`sue_theta`, `elastic_demand`, `combined_demand`, `br_epsilon`, `side_capacities`,
`link_interaction`, `multiclass` each raise `ValueError` naming the field (the engine's
static `assignment()` cannot represent them in a certifiable form). Nonzero `fixed_cost`
(`toll_weight·toll + distance_weight·length`) is refused: the engine HAS a `toll` column,
but its toll/vot time-conversion is unvalidated and a *negative* toll KILLS the host
process, so tolls are refused now (the possible lift is recorded here, not silently run —
unlike marouter, where a toll model is impossible). A link capacity `< 0.1` is refused up
front (the engine's `fmax(0.1, cap)` cost-law clamp would equilibrate it under a different
BPR; a capacity `≤ 1e-4` is also silently dropped). An `sp_calls`-only budget is refused
(no Dijkstra count is exposed); `sp_calls = 0` is disclosed. `multiclass` is the natural
sprint-2 extension (the engine does native multiclass via `mode_type.csv`), refused here
pending a per-class-flow certificate.

## What the certified row MEANS (headline discipline, adr-025/027)

The row is the honest **engine Frank-Wolfe convergence under an exactly-represented cost
law**. Because the compile map is the identity and A2 proves every link's cost matches,
the certified RG at budget B is FW truncation error — and here that truncation is the
engine's own **line-search stall**: the Armijo backtracking collapses to step 0 within a
few iterations (measured: gap frozen from ~iter 40 on Braess, ~iter 100 on Sioux Falls) and
never recovers, so the certified gap floors at ~1e-2…1e-3, far above a converged `bfw`'s
~1e-16 (pinned: `bfw` beats it by orders of magnitude on Braess). The headline names WHOSE
equilibrium and WHICH axis: "DTALite's static assignment certifies RG X at N iterations on
the same BPR the white-box solvers optimize — the first external engine scored on the
power-4 ladder; `bfw`/`algb` beat it on the convergence axis, as expected." The ceiling is
the wheel-engine-as-shipped's line-search stall, **not** the mapping (there is none) and
**not** the benchmark.

## Alternatives considered

- **`path4gmns` (the sounder engine)** — the sibling PyPI package (Peiheng Li + Xuesong
  Zhou) exposes an in-process `find_ue`/`find_ue_fw` with iteration AND target-gap
  parameters and a **returned gap** (a cleaner P6 map than a zero-argument CWD-coupled
  ctypes call), real license text, and active maintenance. It was NOT chosen for sprint 1
  only because `dtalite-adapter` is the named sprint target with the journal anchor
  (`zhou2014dtalite`); `path4gmns` has no journal tool paper of its own (a GitHub citation
  only). Recorded as a **possible future row** — same GMNS dialect and datasets, the same
  honest-sourcing caveat (validate the adapter+engine artifact, not paper numerics).
- **`simulation()` (mesoscopic DTA)** — the engine's separate `DTA_SimulationAPI` is the
  Zhou & Taylor (2014) queue-based within-day simulator (agent/trajectory outputs). None of
  the repo's dynamic certifiers (`dta/`, `dnl/`, `estimation_dynamic.py`) accepts an
  external simulator's trajectories on a static TNTP scenario; wiring one needs a new
  dynamic scenario family + certificate ADR — a **named non-goal** here (the follow-up on
  the DTA/DNL ladder, where the 2014 paper is genuinely on-point).
- **The engine's built-in ODME** (`odme_mode`, Lu/Zhou/Zhang lineage) — a named future
  **T2 guarded estimator**, the `spsa-sumo` (adr-028) analog: the same adapter reused inside
  the estimation track, certified by the unchanged pinned-bfw certifier. Not in sprint 1.
- **In-process ctypes** — disqualified (§4): `ExitMessage` = `getchar()` + `exit()`
  in-process would hang or kill the harness, `assignment()` reads/writes CWD (needing
  process-global `chdir`), and a second call in one process doubles the flows.
- **A new `external`/`static_ue` paradigm token or an Evaluator branch** — none built; the
  `heuristic` paradigm + the existing certificate already score the row honestly, and the
  golden Braess hash is re-asserted byte-identical.

## Honest sourcing (the central section)

- **`zhou2014dtalite` anchors the software LINEAGE, not the formulation** (tool-paper
  discipline, exactly as ADR-027 did for `lopez2018microscopic`). The Cogent Engineering
  2014 paper (read in full: CC-BY 3.0, DOI 10.1080/23311916.2014.961345) describes a
  **mesoscopic queue-based simulator + ODME** and contains **no BPR/VDF anywhere** — its
  "assignment" is gradient-projection path-flow adjustment toward a gap-function DUE,
  specified only by citation (Lu, Mahmassani & Zhou 2009). It reports **no equilibrium
  benchmark numerics** (the Triangle Regional case study is proprietary and un-shipped), so
  like `lopez2018microscopic` it is a **tool-paper anchor**: the row validates adapter +
  engine fidelity, **never the paper's numbers**.
- **What `assignment()` actually is.** The wheel's only C++ is `TAPLite.cpp`, whose header
  states it is "built based on … Bar-Gera's FW.zip". `DTALite.assignment()` →
  `DTA_AssignmentAPI()` is therefore a **classic static link-based Frank-Wolfe TAP loop on
  BPR costs, derived from Bar-Gera's FW.zip** — the 2014 paper's mesoscopic DUE machinery is
  NOT exercised (that is the separate `simulation()` entry). The model card says so
  explicitly. Wheel↔source identity was cross-checked two ways (`nm -D` exports exactly
  `DTA_AssignmentAPI`/`DTA_SimulationAPI`; the `.so`'s format strings match `TAPLite.cpp`),
  but the wheel binary is not provably built from the public source — the behaviors here are
  the executed **wheel**'s, version-scoped to 0.8.1.
- **Artifact provenance.** PyPI `DTALite` 0.8.1 (2025-04-08), authors Fang Tang + Xuesong
  Zhou, home `github.com/itsfangtang/DTALite_release`; the release's own citation line is
  "Tang, F., Zheng, H., and Zhou, X. (2025)". A young package (4 releases). The PyPI
  metadata asserts **Apache 2.0**, but the **`LICENSE` file in the source repo is empty
  (0 bytes)** — GitHub reports SPDX `NOASSERTION`. For CI *consumption* (pip install, run
  the artifact) this is acceptable (the metadata grant is the authors' declaration, nothing
  is copyleft), but the repo vendors **no** engine code and commits **no** sample data (the
  TNTP-derived `data_sets/` are academic-use-only with no license metadata; the harness
  generates its own Braess/two-route GMNS and downloads Sioux Falls checksummed on demand,
  P9). The empty-LICENSE state is recorded here.

## Wall math

One `wall_seconds` deadline is computed once from `solve()` start and threaded through
input-writing → the single engine subprocess → output parsing (only one subprocess phase,
unlike marouter's two). With `OMP_NUM_THREADS = 1` (belt-and-braces with
`number_of_processors = 1` — the engine's default OpenMP thread count is all cores, pure
spin: measured Sioux Falls 8.5 s wall by default vs **0.1 s** at `OMP_NUM_THREADS=1`), a
solve is python startup + `import DTALite` (banner + `.so` load) + `assignment()` + output
write ≈ 0.5 s (Braess/two-route trivially; Sioux Falls 100 FW iterations ~0.1 s CPU). The
CI job: setup ~40 s, install ~20 s (1.3 MB wheel), data prefetch cached, anchors seconds →
**~1.5–3 min**.

## Consequences

- **New:** `DTALiteTapModel` (`dtalite-tap`, registered when `DTALite` present);
  `src/tabench/models/adapters/dtalite_tap.py`; the `dtalite` optional extra; one CI job
  (the fifth); `tests/test_dtalite.py` (26 tests — A1–A4, the near-AON/monotonicity/
  determinism controls, the refusal + sp_calls-only + wall-timeout gates, the read-back
  gate, temp-dir hygiene, the banner-suppression + core-install guard unique to this
  adapter, and the review regressions: the permuted-links sort, the capacity-clamp refusal,
  the BIG-M mass-gate raise, the garbage-cell + wrong-`travel_time` A2 read-back, the
  parse-phase deadline, and the summary torn-pair). No new certificate, scenario field,
  Evaluator branch, or paradigm; no change to `capabilities.py`, `gaps.py`, or any hash.
- **Unchanged:** the Evaluator, the fairness gate, every hash (the golden Braess content
  hash is re-asserted byte-identical in the new test file), and the numpy-only core
  (`import tabench` and the full suite pass without the wheel — the dtalite-free matrix legs
  are the live regression).
- **Follow-ups:** the mesoscopic `simulation()` DTA row (a different equilibrium concept,
  the DTA/DNL track); the built-in ODME as a T2 guarded estimator (the spsa-sumo analog); a
  `path4gmns` static row (the sounder API, no journal anchor); the toll/`vot` lift of the
  `fixed_cost` refusal (once the time-conversion is validated); multiclass via
  `mode_type.csv` — note the read-back/mass gate consume the **pce-weighted** `volume`
  column, which equals `mod_vol_auto` at `pce = 1` (the single-mode case); a multiclass
  lift must switch to the per-mode `mod_vol_<mode>` columns and needs a per-class-flow
  certificate.

## Adversarial review

Three independent lenses (soundness, formulation, numerics), each executing
Python/pytest against the running 0.8.1 wheel; every finding CONFIRMED by a runnable repro
and fixed with a per-finding regression (streak: 18/18 sprints with at least one material
defect; 26 dtalite tests after the fixes, from 19).

**CRITICAL (soundness): an ungrouped `link.csv` corrupts the engine's adjacency.** The
engine builds routing from contiguous `FirstLinkFrom/LastLinkFrom` node ranges, so a
`link.csv` NOT grouped by from-node silently mis-routes: a permuted Braess certified
`feasible=1` at the WRONG flows (RG 0.208 vs 0.0118), and a permuted Sioux Falls sent the
Frank-Wolfe loop into an **infinite loop** — which, under an iterations-only budget
(`timeout=None`), hangs `solve()` forever. FIXED: `_write_gmns` writes links sorted by
`(from_node_id, to_node_id)`; the read-back matches by node pair, so the solve is now
permutation-invariant (pinned by a permuted-Braess-reproduces-the-solve regression).

**MAJORs, all fixed + pinned:** (a) the engine clamps `capacity` at `fmax(0.1, cap)` in
the cost law only (the Beckmann integral unclamped), so a capacity in `(1e-4, 0.1)`
equilibrated under a DIFFERENT BPR with a passing echo read-back (measured A2 rel 0.93,
false "pure FW truncation" provenance) — the refusal threshold is raised to `< 0.1`,
naming the clamp; (b) the ~`1e7` BIG-M cost ceiling silently ZEROES an OD whose congested
cost exceeds it (single link `cap=100 power=4`: `demand=9000` → 9000, `demand=10000` → 0),
laundering to `feasible=0` — a **per-origin mass gate** now RAISES on any origin whose
outflow fails to cover its demand; (c) a corrupt-but-parseable engine output raised a raw
`ValueError`/`TypeError` that escaped `solve()` and collided with the `ValueError` refusal
channel — every cell conversion and the file read are now wrapped to `_ReadBackError` →
the contract `RuntimeError` with stdout+stderr tails.

**MINORs, fixed:** the echo read-back was vacuous below its `atol=1e-3` (a doctored/ignored
sub-1e-3 `vdf_fftt` was accepted) — a **runtime A2 cost-match** (engine `travel_time` ==
repo BPR at emitted flows on every link, rel `1e-3`) is now the live gate, catching both
the fftt drift and the capacity clamp; the wall deadline did not cover the parse phase (a
slow parse overran a 1.5 s budget to 3 s) — one deadline now bounds the read-back too; the
summary parser could pair an `iter` from one line with a stale `gap` from another under
format drift (now commits the two together); the zero-demand path omitted
`engine_configured_iterations` (now emits both keys); the self-gap docstring overstated the
`1/(1−RG)` relation (scoped to the converged regime — it can be negative/frozen off it).

**Survived (highlights):** **Anaheim ftn=39 end-to-end** — the explicit
`first_through_node_id` semantics match TNTP exactly on grouped input (`feasible=1`,
RG 5.94e-5, ZERO through-centroid traffic against a `bfw` detector control); illegal-
shortcut flows cannot certify a negative gap (the Evaluator's nonnegative-excess audit
censors them); the identity compile map re-verified by the A2 cost-match on every link
across Braess, Sioux Falls, a 60-net random adversarial parameter sweep (fft 1e-6..1e3,
b 0..1e7, power {0, 1, 2.5, 4, 6.87}, zero law mismatches), **Winnipeg** (2836 links,
powers 0–6.87, RG 1.176e-3) and **Barcelona** (2522 links, powers to 16.83, RG 5.48e-4) —
the first big-net exposure, both `feasible=1` and A2 ≤ 7.6e-4 under the `1e-3` gate;
byte-determinism across reruns and under a hostile parent `OMP_NUM_THREADS=64` (the
child override wins); 8-way parallel solves with no cross-talk; the mid-engine wall kill
(no leaked tempdirs, no zombie children); the banner-suppression + core-install guard
(blocked-`DTALite` unregisters the model, `import tabench` stdout empty); the
`ValueError`-refusal vs `RuntimeError`-crash separation; CI install-order simulation in
fresh venvs (pin-first, no re-resolution; the core leg collects 731 tests without the
wheel); temp-dir hygiene across the normal/refusal/raise/timeout paths; the golden
Braess content hash re-asserted byte-identical.
