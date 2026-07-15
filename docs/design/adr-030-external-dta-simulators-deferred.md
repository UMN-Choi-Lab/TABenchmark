# ADR-030 — MATSim / DynaMIT / DYNASMART adapters: measured deferral, and the ADR that unblocks them

**Status:** accepted (deferral record)
**File:** `docs/design/adr-030-external-dta-simulators-deferred.md`

## Context — the queue after `dtalite-tap`, and a rule earned twice

The external-adapter queue holds three more rows: `matsim-adapter` (Horni et al. 2016,
canon `horni2016multiagent`), `dynamit-adapter` (Ben-Akiva et al. 2001), and the
DYNASMART pair (Jayakrishnan et al. 1994 adapter / Peeta & Mahmassani 1995 solver). The
"external tools cannot run in CI" assumption has now fallen **twice** — `eclipse-sumo`
(ADR-027) and `DTALite` (ADR-029) both ship engine binaries in pip wheels — so no row may
be deferred on speculation: this ADR records **measured probes, run 2026-07-15**, for all
three, and defers them on what the probes actually found. A deferral here is an attempt
record, not a guess.

## Decision — DEFER all three adapter rows; the blocker is named per row

### MATSim: executable, but the certificate it needs does not exist yet

**Executability is PROVEN, not presumed** (probe record, this box, 2026-07-15):
PyPI `install-jdk` fetches a Temurin JDK without root (~6 s; MATSim 2026.0 needs JDK 25,
2025.0 runs on JDK 21 — both verified); the official `matsim-2026.0-release.zip`
(113 MB, GitHub `matsim-org/matsim-libs`) downloaded in ~2 s; a hand-built two-route
scenario (6 nodes / 7 links — the engine forces entry/exit links and strong connectivity,
both crash surfaces captured) with 100 agents ran 11 iterations headless in **6.6 s /
550 MB RSS**; per-link flows come from `output_links.csv.gz` and events (caveat: the CSV
undercounts the arrival link — count `entered link` events); **same seed → bit-identical
outputs**, even across the 2025.0/2026.0 releases; exit codes are trustworthy (rc 1 on
all six failure probes, informative messages). A CI job (actions/setup-java + cached zip)
projects to 1–2 min cold. The PyPI `matsim` package (0.2.3) is an unrelated *neuronal*
simulator (name collision) and `matsim-tools` is file I/O only — the engine is Java-only.
(Full recipe: the scoping dossiers, session scratchpad `matsim-probe/`.)

**The blocker is formulation — A2 is impossible in kind, not degree.** Both shipped
external-engine ADRs made the cost-matched anchor the shipping condition ("the row does
not ship without A2", ADR-027/029). MATSim's QSim has **no static latency function**:
each link's traversal time is `length/freespeed` — a **constant** — whenever the flow
capacity does not bind, and emergent time-varying queueing when it does. No
`(freespeed, flowCapacity, storageCapacity)` choice produces `t = fft·(1 + b·(v/c)^p)`
on any interior flow range; the exactly-representable set under the declared BPR is
**`{b = 0}`** — constant-cost links, which `aon` already matches trivially. The concrete
consequence (analytic, from the declared Braess costs — NOT a measured MATSim run): with
capacities above saturation every link is cost-flat below capacity, so MATSim's
co-evolution converges — correctly, under ITS OWN model — to all demand on the bypass
`(6,0,6,0,6)`, which the harness certifies at **RG = (816−660)/816 ≈ 0.19**. That number
would conflate four unattributable error sources (queue-vs-BPR cost-law gap; the invented
time axis a static scenario doesn't have; integer-agent quantization; seeded replanning
noise) with **no decomposition gate** — unlike marouter's `incremental` negative control
(RG 0.07, attributed to a non-equilibrium loading of a *matched* cost law). The
equilibrium concept itself differs: a stochastic co-evolutionary fixed point over plans
under experienced dynamic times, not Wardrop UE under static declared costs — the same
category ADR-027 rejected for `duaIterate` and ADR-029 named a non-goal for DTALite's
`simulation()`. `docs/ARCHITECTURE.md` already states the policy: external engines get
"certification where static costs permit; **otherwise scored on the observational
track**" — and that observational track for external dynamic engines does not exist yet.

**Scoped-ship variants considered and rejected:** (a) a `{b=0}`-only slice duplicates
`aon`; (b) a no-certificate P5 observational row violates the A2 shipping bar and the
duaIterate precedent (a different equilibrium concept is not scored against static BPR
at all); (c) calibrating MATSim capacities per-scenario until Braess flows match the UE
oracle inverts the benchmark's direction (the harness certifies models; models are never
tuned to the reference) — refused.

### DynaMIT: no public artifact exists (probe record)

GitHub search returns **zero** public DynaMIT repositories (re-verified independently);
the system was historically distributed by the MIT ITS Lab on request; the successor
SimMobility (`smart-fm/simmobility-prod`) is public but carries SPDX `NOASSERTION`
licensing. There is nothing to adapt that a public benchmark may execute or that CI may
install. Deferred on artifact absence.

### DYNASMART: licensing bars a CI row; the WHITE-BOX sibling stays live

FHWA DYNASMART-P is distributed under FHWA/McTrans licensing (no pip/public artifact;
the McTrans distribution page redirects). A licensed binary cannot back a public CI row.
The **adapter** path is deferred — but the TASKS queue item bundles it with
**Peeta & Mahmassani (1995)** (*System optimal and user equilibrium time-dependent
traffic assignment in congested networks*), a **white-box solver** row on the roadmap
that needs no binary at all: the SO/UE time-dependent formulation can be implemented
from the paper on the repo's own DTA ladder (the `merchant-nemhauser`/`lp-so-dta`
precedent). That row is NOT deferred by this ADR; it stays in the implementable queue.

## What would unblock the adapters (named, concrete)

**One ADR: the external-dynamic-engine observational certificate** — the dynamic
scenario family + certificate for external-simulator output that ADR-027 (the
`duaIterate` follow-up) and ADR-029 (the `simulation()` follow-up) both name. It
unblocks **three queued rows at once** — MATSim, SUMO `duaIterate`, DTALite
`simulation()` — a far better sprint shape than forcing any one of them through the
static certificate. The repo's own DNL certificates (C0–C8, ADR-010 and the gated
review) are the natural scoring substrate. On that ladder MATSim ships as the **first
agent-based, first stochastic-track external engine** (`global.randomSeed` → seedable,
P8 macroreps + bootstrap CIs), with the execution recipe above already proven on this
box. DynaMIT remains artifact-blocked regardless; DYNASMART remains license-blocked
regardless — those two need upstream changes no repo ADR can supply.

## Consequences

- No code, no new dependency, no CI job, no hash change — a documentation-only record.
- `docs/ROADMAP.md`: the Horni et al. (2016) and Ben-Akiva et al. (2001) black-box rows
  and the Jayakrishnan et al. (1994) evaluation-tool row carry a deferral pointer to
  this ADR (hand-annotated, like shipped flips — `tools/generate_references.py` is never
  run). The Peeta & Mahmassani (1995) white-box row is untouched and live.
- `docs/ARCHITECTURE.md`: the external-engines paragraph's "MATSim planned" updated to
  point here.
- The scoping dossiers (execution + formulation, with the full probe logs) live in the
  session scratchpad; their load-bearing facts are recorded above so the deferral
  survives the scratchpad.
- Better next sprints, per the same scoping: `xu2024-dataset` (CC-BY download-on-demand
  fetcher, pure P9), `simopt-protocol` (progress curves / solvability profiles, pure
  metric code serving P5), `bo4mob-scenario` (arXiv 2510.18824) — each produces
  certified, interpretable artifacts now.
