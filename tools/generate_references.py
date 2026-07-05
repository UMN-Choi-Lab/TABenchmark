"""Regenerate docs/REFERENCES.md and docs/ROADMAP.md from docs/references.json.

The canonical, verified reference data lives in ``docs/references.json``
(with BibTeX in ``docs/references.bib``). The two markdown documents are
generated views — edit the JSON (and BibTeX), then rerun:

    python tools/generate_references.py
"""

from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path

DOCS = Path(__file__).resolve().parent.parent / "docs"

FAMILY_TITLES = OrderedDict(
    [
        ("foundations", "Foundations of Static Equilibrium Assignment"),
        ("link-based-algorithms", "Link-Based UE Algorithms"),
        ("path-and-bush-based", "Path-Based and Bush/Origin-Based UE Algorithms"),
        ("stochastic-ue", "Stochastic User Equilibrium and Route Choice"),
        ("so-and-pricing", "System Optimum, Congestion Pricing, and Efficiency"),
        ("extensions-static", "Static Assignment Extensions"),
        ("dta-analytical", "Analytical Dynamic Traffic Assignment"),
        ("dnl-models", "Dynamic Network Loading Models"),
        ("dta-simulation-software", "Simulation-Based DTA and Software Systems"),
        ("day-to-day", "Day-to-Day Dynamics and Learning Processes"),
        ("ml-based-ta", "Machine-Learning-Based Traffic Assignment"),
        (
            "data-calibration-benchmarks",
            "Data, OD Estimation, Calibration, and Benchmarking Practice",
        ),
    ]
)

ROADMAP_TITLES = {
    "foundations": "Foundations",
    "link-based-algorithms": "Link-based UE algorithms",
    "path-and-bush-based": "Path/bush-based UE algorithms",
    "stochastic-ue": "Stochastic UE & route choice",
    "so-and-pricing": "System optimum & pricing",
    "extensions-static": "Static extensions",
    "dta-analytical": "Analytical DTA",
    "dnl-models": "Dynamic network loading",
    "dta-simulation-software": "Simulation-based DTA & software",
    "day-to-day": "Day-to-day dynamics",
    "ml-based-ta": "ML-based traffic assignment",
    "data-calibration-benchmarks": "Data, estimation & benchmarking",
}

ROADMAP_VERSION = {
    "foundations": "v0-v1 (formulations underpin metrics)",
    "link-based-algorithms": "v0 (FW, MSA) / v0.x (CFW, BFW)",
    "path-and-bush-based": "v1",
    "stochastic-ue": "v0.x (Dial, logit-SUE MSA) / v1 (probit)",
    "so-and-pricing": "v1",
    "extensions-static": "v1",
    "dta-analytical": "v2",
    "dnl-models": "v2",
    "dta-simulation-software": "v2 (adapters)",
    "day-to-day": "v2",
    "ml-based-ta": "v1 (baseline wrappers)",
    "data-calibration-benchmarks": "v1 (T2 estimation track)",
}

ROLE_SHORT = {
    "white-box solver": "white-box solver",
    "black-box wrapper": "black-box wrapper",
    "network-loading component": "network loading",
    "route-choice component": "route choice",
    "data/scenario": "data/scenario",
    "metric/protocol": "metric/protocol",
    "survey/context": "survey/context",
}

VERDICT_MARK = {
    "verified": "✓",
    "verified-manual": "✓ (manual)",
    "book-manual": "book",
    "partial_match": "partial",
}

# bibkeys whose method (or formulation) ships, as (version, description)
SHIPPED = {
    "frank1956algorithm": ("v0", "Frank-Wolfe solver"),
    "leblanc1975efficient": ("v0", "Frank-Wolfe solver"),
    "wardrop1952some": ("v0", "equilibrium conditions in the certified gap"),
    "beckmann1956studies": ("v0", "Beckmann objective in metrics"),
    "bpr1964traffic": ("v0", "BPR link performance function (Network.link_cost)"),
    "mitradjieva2013stiff": ("v0.x", "conjugate and bi-conjugate FW solvers"),
    "boyce2004convergence": ("v0.x", "convergence target protocol (Budget.target_relative_gap)"),
    "dial1971probabilistic": ("v0.x", "STOCH loading map (models/_stoch.py)"),
    "fisk1980some": ("v0.x", "logit SUE task (fixed-point certificate, ADR-001)"),
    "powell1982convergence": ("v0.x", "MSA-SUE solver step sizes"),
    "stabler2016transportation": ("v0.x", "checksummed TNTP fetcher + 4 registered networks"),
    "jayakrishnan1994faster": ("v1", "path-based gradient projection solver (gp)"),
    "yang1998principle": ("v1", "first-best marginal-cost tolls (metrics.so)"),
    "roughgarden2002how": ("v1", "price-of-anarchy protocol + certified SO gap"),
    "vanzuylen1980most": ("v1", "T2 entropy estimator (vzw-entropy, ADR-002)"),
    "cascetta1984estimation": ("v1", "T2 GLS estimator (gls, ADR-002)"),
    "spiess1990gradient": ("v1", "T2 gradient OD adjustment (spiess, ADR-002)"),
    "spall1992multivariate": ("v1", "T2 SPSA calibration baseline (spsa, ADR-002)"),
    "hazelton2015network": ("v1", "T2 identifiability report + held-out ranking protocol"),
    "dial2006path": ("v1", "Algorithm B bush-based solver (algb)"),
    "sheffi1982algorithm": ("v1", "probit SUE solver (sue-probit-msa) + MC certificate (ADR-003)"),
    "daganzo1977stochastic": ("v1", "SUE definition underlying the probit task"),
}


def _authors_short(entry: dict) -> str:
    authors = entry.get("authors") or ["?"]
    if len(authors) == 1:
        return authors[0]
    if len(authors) == 2:
        return f"{authors[0]} & {authors[1]}"
    return f"{authors[0]} et al."


def _by_primary_family(refs: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {key: [] for key in FAMILY_TITLES}
    for entry in refs:
        families = entry.get("families") or []
        primary = next((f for f in FAMILY_TITLES if f in families), None)
        if primary is None:
            raise ValueError(f"{entry.get('bibkey')}: no recognized family in {families}")
        grouped[primary].append(entry)
    return grouped


def write_references_md(refs: list[dict]) -> None:
    grouped = _by_primary_family(refs)
    n_by_tier = {t: sum(1 for e in refs if e.get("tier") == t) for t in (1, 2, 3)}

    lines = [
        "# The TABenchmark Reference Canon",
        "",
        "The models, algorithms, data practices, and protocols that a complete traffic",
        "assignment benchmark must cover — compiled family-by-family across ~50 years of",
        "transportation research and **verified reference-by-reference** against Crossref /",
        "Semantic Scholar (via refcheck). BibTeX for every entry is in",
        "[`references.bib`](references.bib); the machine-readable canon is",
        "[`references.json`](references.json).",
        "",
        "**Tiers** — 1: must implement in the benchmark core; 2: should implement as a",
        "variant/extension; 3: background, survey, or book (cite, don't implement).",
        "**Verified** — ✓: metadata verified against publication databases; *book*:",
        "hand-checked @book entry; *partial*: verified after correcting the originally",
        "compiled metadata (corrections are recorded in the entry's `contribution` field",
        "in `references.json`).",
        "",
        f"**{len(refs)} references** — {n_by_tier[1]} tier-1, {n_by_tier[2]} tier-2, "
        f"{n_by_tier[3]} tier-3.",
    ]

    for family, title in FAMILY_TITLES.items():
        entries = sorted(
            grouped[family], key=lambda e: ((e.get("tier") or 9), e.get("year") or 0)
        )
        if not entries:
            continue
        lines += [
            "",
            f"## {title}",
            "",
            "| Reference | Title | Venue | Tier | Role | Verified |",
            "|---|---|---|---|---|---|",
        ]
        for e in entries:
            ref = f"{_authors_short(e)} ({e.get('year', '?')})"
            title_txt = (e.get("title") or "").replace("|", "\\|")
            if e.get("doi"):
                title_txt = f"[{title_txt}](https://doi.org/{e['doi']})"
            venue = (e.get("venue") or "").replace("|", "\\|")
            role = ROLE_SHORT.get(e.get("implement_as"), e.get("implement_as") or "–")
            mark = VERDICT_MARK.get(e.get("verdict"), e.get("verdict") or "?")
            lines.append(
                f"| {ref} | {title_txt} | {venue} | {e.get('tier') or '–'} | {role} | {mark} |"
            )

    lines += [
        "",
        "---",
        "",
        "*Compiled by a 12-family literature sweep with per-reference verification,",
        "then extended by a citation-graph completeness sweep (forward + backward",
        "citations of every canon entry via OpenAlex, plus per-family keyword",
        "searches; candidates filtered by skeptical judging and verified against",
        "Crossref). 0 references failed verification. Cross-family duplicates were",
        "merged (a reference appears in the family where it is most load-bearing;",
        "`references.json` records all its families). Generated by",
        "`tools/generate_references.py` — edit the JSON, not this file.*",
        "",
    ]
    (DOCS / "REFERENCES.md").write_text("\n".join(lines))


def write_roadmap_md(refs: list[dict]) -> None:
    grouped = _by_primary_family(refs)
    n_tier1 = sum(1 for e in refs if e.get("tier") == 1)

    lines = [
        "# Implementation Roadmap",
        "",
        f"The tier-1 canon ({n_tier1} must-implement references from",
        "[REFERENCES.md](REFERENCES.md)), staged by version. Checked items ship in the",
        "current release (v0 also ships MSA and all-or-nothing as baselines).",
        "",
        "Version staging: **v0** core harness + link-based solvers -> **v0.x** accelerated",
        "FW, logit SUE, Anaheim/Barcelona/Winnipeg rungs (this release; plugin registry and",
        "profiles still open) -> **v1** bush-based solvers, SUE variants, static extensions,",
        "T2 estimation track -> **v2** DTA, network loading, engine adapters, day-to-day,",
        "T3 interventions.",
    ]

    for family in FAMILY_TITLES:
        tier1 = sorted(
            [e for e in grouped[family] if e.get("tier") == 1],
            key=lambda e: e.get("year") or 0,
        )
        if not tier1:
            continue
        lines += ["", f"## {ROADMAP_TITLES[family]} — {ROADMAP_VERSION[family]}", ""]
        for e in tier1:
            shipped = SHIPPED.get(e.get("bibkey"))
            mark = "x" if shipped else " "
            suffix = f" — **shipped in {shipped[0]}** ({shipped[1]})" if shipped else ""
            role = e.get("implement_as") or ""
            lines.append(
                f"- [{mark}] {_authors_short(e)} ({e.get('year', '?')}) — "
                f"*{e.get('title')}* ({role}){suffix}"
            )

    lines += [
        "",
        "---",
        "*Generated from the verified canon `references.json` by",
        "`tools/generate_references.py`; regenerate rather than hand-edit.*",
        "",
    ]
    (DOCS / "ROADMAP.md").write_text("\n".join(lines))


def main() -> None:
    refs = json.loads((DOCS / "references.json").read_text())
    write_references_md(refs)
    write_roadmap_md(refs)
    print(f"Regenerated REFERENCES.md and ROADMAP.md from {len(refs)} references.")


if __name__ == "__main__":
    main()
