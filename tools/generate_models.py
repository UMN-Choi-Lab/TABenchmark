"""Regenerate docs/MODELS.md, docs/model-evolution.dot, and docs/model-evolution.svg
from docs/model-specs.json (the grounded, verified model spec DB).

The spec DB is the canonical source; the three documents are generated views. Edit
the JSON (adding models as they ship, flipping `shipped`/`tabench_name`), then rerun:

    python tools/generate_models.py

Each model's `evolved_from` list (bibkeys of direct predecessors) defines the edges of
the evolution graph. Companion: docs/MODELS.md is the prose compendium; the .dot/.svg are
the lineage graph (`dot -Tsvg model-evolution.dot` if graphviz is installed; the .svg is
also generated directly here so no graphviz is required).
"""

from __future__ import annotations

import html
import json
import re
import zlib
from pathlib import Path

DOCS = Path(__file__).resolve().parent.parent / "docs"

FAM_ORDER = [
    ("foundations", "Foundations — the principles and the convex program"),
    ("link-based-algorithms", "Link-based UE algorithms — the convergence race begins"),
    ("path-and-bush-based", "Path- and bush-based UE algorithms — high-precision equilibria"),
    ("stochastic-ue", "Stochastic user equilibrium — perception and route choice"),
    ("so-and-pricing", "System optimum & pricing — the price of selfishness"),
    ("extensions-static", "Static extensions — relaxing the fixed-demand assumption"),
    ("day-to-day", "Day-to-day dynamics — how equilibrium is reached (or not)"),
    ("dnl-models", "Dynamic network loading — putting time into the links"),
    ("dta-analytical", "Analytical dynamic traffic assignment — equilibrium over time"),
    ("ml-based-ta", "Machine-learning traffic assignment — assignment as a function"),
    ("data-calibration-benchmarks", "Data, OD estimation & benchmarking — the inverse problem"),
]
PALETTE = {
    "foundations": "#e8e8e8", "link-based-algorithms": "#d6e4ff",
    "path-and-bush-based": "#c9f0d0", "stochastic-ue": "#ffe9c7",
    "so-and-pricing": "#ffd6d6", "extensions-static": "#e6d6ff",
    "day-to-day": "#d6f5f0", "dnl-models": "#f5e6d6", "dta-analytical": "#f0d6e6",
    "ml-based-ta": "#d6f0ff", "data-calibration-benchmarks": "#eeeed6",
}

INTRO = """# The TABenchmark Model Compendium

*What every model in the benchmark is, and — the point of this document — what each one \
does **differently** from what came before.* Fifty years of traffic assignment is not a \
pile of interchangeable solvers; it is an evolutionary tree, each node answering a \
shortcoming of its parent. This compendium is the field guide; the \
[evolution graph](#the-evolution-of-traffic-assignment) is the map.

Every model is read through TABenchmark's **P1 lens**: the harness recomputes the scored \
metric from emitted link flows, so "what it does differently" is always graded on the \
same certificate (see [ARCHITECTURE.md](ARCHITECTURE.md), [VALIDATION.md](VALIDATION.md)). \
Entries marked **shipped** run today; **roadmap** entries are queued in \
[ROADMAP.md](ROADMAP.md) / `TASKS.md`. Grounded in the verified \
[reference canon](REFERENCES.md) and the Boyles/Lownes/Unnikrishnan *Transportation \
Network Analysis* text.

## The evolution of traffic assignment

The trunk runs **principle → program → cost → solver**: Wardrop (1952) stated *what* \
equilibrium is, Beckmann (1956) turned it into a convex program *solvable* in principle, \
the BPR curve (1964) gave it a differentiable cost, and Frank–Wolfe/LeBlanc (1956/1975) \
made it *computable*. Everything else is a branch off that trunk, each driven by a \
specific limitation:

- **the convergence race** (link → conjugate → path → bush): Frank–Wolfe's slow tail \
drove conjugate FW, then path-based gradient projection, then the origin/bush-based \
family (OBA → Algorithm B → TAPAS) that reaches machine-precision equilibria.
- **perception** (stochastic UE): drivers don't see true costs, so Dial's STOCH, Fisk's \
logit, and the Daganzo–Sheffi/Sheffi–Powell probit line replaced the sharp min with a \
choice model.
- **the planner's view** (SO & pricing): Beckmann's marginal-cost transform, Yang–Huang \
tolls, and Roughgarden–Tardos' price of anarchy quantified the cost of selfish routing.
- **relaxing assumptions** (static extensions): elastic demand (Florian–Nguyen), combined \
distribution+assignment (Evans), bounded rationality, and side constraints loosened the \
fixed-demand, perfectly-rational, uncapacitated idealization.
- **adding time** (dynamics): day-to-day systems (Smith, Horowitz, Cascetta) asked whether \
equilibrium is even reached; dynamic network loading (LWR → CTM → LTM → node models) and \
analytical DTA (Vickrey → Merchant–Nemhauser → Friesz → Ziliaskopoulos) put congestion in \
motion.
- **assignment as a function** (learned): Rahman–Hasan regress flows, Liu et al. enforce \
the fixed point inside the network, Liu–Meidani generalize across topologies — trading \
the equilibrium guarantee for speed, which P1 is built to audit.
"""


def short(m):
    return f'{m["authors_short"].split(" & ")[0].split(" et al")[0].split(",")[0]} {m["year"]}'


def node_id(bibkey):
    return "n_" + re.sub(r"[^a-z]", "", bibkey.lower())[:22] + str(zlib.crc32(bibkey.encode()) % 10000)


def build_mermaid(models, nid):
    fam_cls = {f: "c" + str(i) for i, f in enumerate(PALETTE)}
    out = ["graph TD"]
    for m in sorted(models, key=lambda x: x.get("year") or 0):
        sl, sr = ("([", "])") if m.get("shipped") else ("[", "]")
        out.append(f'  {nid[m["bibkey"]]}{sl}"{short(m)}"{sr}:::{fam_cls[m["_family"]]}')
    for m in models:
        for p in m.get("evolved_from") or []:
            if p in nid:
                out.append(f'  {nid[p]} --> {nid[m["bibkey"]]}')
    for f, c in fam_cls.items():
        out.append(f"  classDef {c} fill:{PALETTE[f]},stroke:#333,color:#000;")
    return "\n".join(out)


def build_dot(models, nid):
    out = [
        "digraph TA_evolution {",
        '  rankdir=TB; node [shape=box,style="rounded,filled",fontname="Helvetica",fontsize=10];',
        '  edge [color="#666666",arrowsize=0.7];',
    ]
    for y in sorted({m.get("year") or 0 for m in models}):
        same = [f'"{m["bibkey"]}"' for m in models if (m.get("year") or 0) == y]
        out.append(f'  {{ rank=same; {" ".join(same)} }}')
    for m in models:
        style = '"rounded,filled"' if m.get("shipped") else '"rounded,filled,dashed"'
        label = f'{short(m)}\\n{m.get("tabench_name") or ""}'.strip()
        out.append(f'  "{m["bibkey"]}" [label="{label}",fillcolor="{PALETTE[m["_family"]]}",style={style}];')
    for m in models:
        for p in m.get("evolved_from") or []:
            if p in nid:
                out.append(f'  "{p}" -> "{m["bibkey"]}";')
    out.append("}")
    return "\n".join(out)


def build_svg(models):
    lanes = list(PALETTE)
    lane_w, top, bot, margin, ys = 250, 70, 40, 20, 22
    y0 = min(m.get("year") or 2100 for m in models)
    y1 = max(m.get("year") or 0 for m in models)

    def yc(yr):
        return top + (yr - y0) * ys

    height = yc(y1) + bot + 40
    width = margin * 2 + lane_w * len(lanes)
    pos, buckets = {}, {}
    for m in sorted(models, key=lambda x: (x.get("year") or 0, x["bibkey"])):
        lane = lanes.index(m["_family"])
        key = (lane, m.get("year") or 0)
        n = buckets.get(key, 0)
        buckets[key] = n + 1
        pos[m["bibkey"]] = (margin + lane * lane_w + 24 + (n % 2) * 112, yc(m.get("year") or y0) + (n // 2) * 17)
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" font-family="Helvetica,Arial,sans-serif">']
    s.append(f'<rect width="{width}" height="{height}" fill="#fafafa"/>')
    for i, f in enumerate(lanes):
        lx = margin + i * lane_w
        s.append(f'<rect x="{lx}" y="0" width="{lane_w}" height="{height}" fill="{PALETTE[f]}" opacity="0.18"/>')
        s.append(f'<text x="{lx + lane_w / 2}" y="24" text-anchor="middle" font-size="12" font-weight="bold" fill="#222">{html.escape(f)}</text>')
    for yr in range((y0 // 10) * 10, y1 + 1, 10):
        if yr < y0:
            continue
        yy = yc(yr)
        s.append(f'<line x1="0" y1="{yy}" x2="{width}" y2="{yy}" stroke="#ddd"/>')
        s.append(f'<text x="4" y="{yy - 2}" font-size="10" fill="#999">{yr}</text>')
    s.append('<defs><marker id="a" markerWidth="7" markerHeight="7" refX="6" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6 Z" fill="#888"/></marker></defs>')
    for m in models:
        x2, y2 = pos[m["bibkey"]]
        for p in m.get("evolved_from") or []:
            if p in pos:
                x1, y1e = pos[p]
                s.append(f'<path d="M{x1},{y1e + 7} C{x1},{(y1e + y2) / 2} {x2},{(y1e + y2) / 2} {x2},{y2 - 7}" fill="none" stroke="#bbb" stroke-width="1" marker-end="url(#a)" opacity="0.7"/>')
    for m in models:
        x, y = pos[m["bibkey"]]
        dash = "" if m.get("shipped") else ' stroke-dasharray="3,2"'
        s.append(f'<rect x="{x - 52}" y="{y - 8}" width="104" height="17" rx="4" fill="{PALETTE[m["_family"]]}" stroke="#333"{dash}/>')
        s.append(f'<text x="{x}" y="{y + 4}" text-anchor="middle" font-size="9.5" fill="#000">{html.escape(short(m))}</text>')
    s.append("</svg>")
    return "\n".join(s)


def build_models_md(models, nid, mermaid):
    keys = {m["bibkey"] for m in models}
    L = [INTRO, "```mermaid\n" + mermaid + "\n```\n",
         "*Full Graphviz source: [`model-evolution.dot`](model-evolution.dot); standalone "
         "[`model-evolution.svg`](model-evolution.svg). Rounded/solid nodes = shipped; "
         "square/dashed = roadmap. Edges point parent → descendant.*\n"]
    by_fam = {f: [] for f, _ in FAM_ORDER}
    for m in models:
        by_fam[m["_family"]].append(m)
    for famkey, famtitle in FAM_ORDER:
        fam = sorted(by_fam[famkey], key=lambda x: (x.get("year") or 0, x["bibkey"]))
        if not fam:
            continue
        L.append(f"## {famtitle}\n")
        for m in fam:
            if m.get("shipped") and m.get("tabench_name"):
                ship = f'`{m["tabench_name"]}` · **shipped**'
            elif m.get("shipped"):
                ship = "**shipped**"
            else:
                ship = "_roadmap_"
            L.append(f'### {m["authors_short"]} ({m["year"]}) — {m.get("title", "").strip()}\n')
            L.append(f'{ship} · {m.get("equilibrium_principle", "")} · `[{m["bibkey"]}]`\n')
            if m.get("one_line"):
                L.append(m["one_line"] + "\n")
            if m.get("innovation"):
                L.append(f'**What it does differently.** {m["innovation"]}\n')
            if m.get("math_sketch"):
                L.append(f'**Formulation.** `{m["math_sketch"]}`\n')
            if m.get("validation"):
                L.append(f'**Validation.** {m["validation"]}\n')
            preds = [p for p in (m.get("evolved_from") or []) if p in keys]
            if preds:
                names = ", ".join(f'{next(x for x in models if x["bibkey"] == p)["authors_short"]} '
                                  f'{next(x for x in models if x["bibkey"] == p)["year"]}' for p in preds)
                L.append(f"*Builds on:* {names}.\n")
    return "\n".join(L)


def main():
    models = json.loads((DOCS / "model-specs.json").read_text())
    nid = {m["bibkey"]: node_id(m["bibkey"]) for m in models}
    mermaid = build_mermaid(models, nid)
    (DOCS / "MODELS.md").write_text(build_models_md(models, nid, mermaid))
    (DOCS / "model-evolution.dot").write_text(build_dot(models, nid) + "\n")
    (DOCS / "model-evolution.svg").write_text(build_svg(models) + "\n")
    n_ship = sum(1 for m in models if m.get("shipped"))
    print(f"Regenerated MODELS.md + model-evolution.{{dot,svg}} from {len(models)} models "
          f"({n_ship} shipped, {len(models) - n_ship} roadmap).")


if __name__ == "__main__":
    main()
