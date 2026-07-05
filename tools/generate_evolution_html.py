# -*- coding: utf-8 -*-
"""Regenerate docs/model-evolution.html (self-contained interactive lineage graph)
from docs/model-specs.json. Companion to tools/generate_models.py.

    python tools/generate_evolution_html.py
"""
import json
from pathlib import Path

DOCS = Path(__file__).resolve().parent.parent / "docs"


def _short(m):
    return f'{m["authors_short"].split(" & ")[0].split(" et al")[0].split(",")[0]} {m["year"]}'


_specs = json.loads((DOCS / "model-specs.json").read_text())
DATA = json.dumps([{
    "k": m["bibkey"], "label": _short(m), "auth": m["authors_short"], "year": m["year"],
    "title": m.get("title", ""), "fam": m["_family"], "ship": bool(m.get("shipped")),
    "tab": m.get("tabench_name") or "", "from": list(m.get("evolved_from") or []),
    "eq": m.get("equilibrium_principle", ""), "one": m.get("one_line", ""),
    "innov": m.get("innovation", ""),
} for m in _specs], ensure_ascii=False)

FAM = {
 "foundations":            ("#c7ced8", "Foundations", "principles & the convex program"),
 "link-based-algorithms":  ("#a9c7f5", "Link-based UE", "Frank–Wolfe & the convergence race"),
 "path-and-bush-based":    ("#9fe0ae", "Path / bush UE", "high-precision equilibria"),
 "stochastic-ue":          ("#f5cf8f", "Stochastic UE", "perception & route choice"),
 "so-and-pricing":         ("#f2a9a9", "SO & pricing", "the price of selfishness"),
 "extensions-static":      ("#cdb0f0", "Static extensions", "relaxing fixed demand"),
 "day-to-day":             ("#9fe6da", "Day-to-day", "reaching equilibrium (or not)"),
 "dnl-models":             ("#eccaa2", "Network loading", "putting time in the links"),
 "dta-analytical":         ("#e6a9cd", "Analytical DTA", "equilibrium over time"),
 "ml-based-ta":            ("#a9dcf5", "Learned models", "assignment as a function"),
 "data-calibration-benchmarks": ("#dcdca6", "Estimation & data", "the inverse problem"),
}
fam_js = json.dumps({k:{"c":v[0],"name":v[1],"desc":v[2]} for k,v in FAM.items()}, ensure_ascii=False)
fam_order = json.dumps(list(FAM.keys()))

HTML = r"""<title>The Evolution of Traffic Assignment Models</title>
<style>
  :root{
    --paper:#f4f6f9; --panel:#ffffff; --ink:#16202e; --slate:#5a6478; --slate2:#828da2;
    --hair:#e2e6ee; --hair2:#eef1f6; --accent:#0f6e78; --accent-2:#0b525a;
    --shadow:0 1px 2px rgba(22,32,46,.06),0 8px 24px rgba(22,32,46,.06);
    --serif:"Charter","Iowan Old Style","Palatino Linotype",Georgia,serif;
    --sans:system-ui,-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
  }
  *{box-sizing:border-box}
  body{margin:0;background:var(--paper);color:var(--ink);font-family:var(--sans);
    line-height:1.55;-webkit-font-smoothing:antialiased}
  .wrap{max-width:1400px;margin:0 auto;padding:clamp(20px,4vw,44px) clamp(16px,3vw,32px)}
  header .eyebrow{font-size:12px;letter-spacing:.16em;text-transform:uppercase;
    color:var(--accent-2);font-weight:600;margin:0 0 10px}
  h1{font-family:var(--serif);font-weight:600;font-size:clamp(30px,5vw,50px);line-height:1.05;
    letter-spacing:-.01em;margin:0;text-wrap:balance;max-width:20ch}
  .thesis{font-family:var(--serif);font-size:clamp(16px,2vw,20px);color:var(--slate);
    max-width:60ch;margin:16px 0 0;line-height:1.5}
  .thesis b{color:var(--ink);font-weight:600}
  .chips{display:flex;flex-wrap:wrap;gap:10px;margin:24px 0 0}
  .chip{background:var(--panel);border:1px solid var(--hair);border-radius:10px;
    padding:9px 14px;box-shadow:var(--shadow);display:flex;align-items:baseline;gap:8px}
  .chip b{font-family:var(--serif);font-size:20px;font-variant-numeric:tabular-nums}
  .chip span{font-size:12px;color:var(--slate);letter-spacing:.02em}
  .work{display:grid;grid-template-columns:minmax(0,1fr) 340px;gap:22px;margin-top:30px;align-items:start}
  @media(max-width:960px){.work{grid-template-columns:1fr}}
  .panel{background:var(--panel);border:1px solid var(--hair);border-radius:14px;box-shadow:var(--shadow)}
  .graphhead{display:flex;justify-content:space-between;align-items:center;gap:12px;
    padding:14px 18px;border-bottom:1px solid var(--hair2)}
  .graphhead h2{font-family:var(--serif);font-size:16px;margin:0;font-weight:600}
  .graphhead .hint{font-size:12px;color:var(--slate2)}
  .scroll{overflow:auto;max-height:74vh;border-radius:0 0 14px 14px}
  svg{display:block}
  .node rect{transition:opacity .18s,stroke-width .18s}
  .node text{pointer-events:none;font-family:var(--sans)}
  .edge{transition:opacity .18s,stroke .18s,stroke-width .18s}
  .node{cursor:pointer}
  .dim{opacity:.16}
  .rail{position:sticky;top:18px;display:flex;flex-direction:column;gap:14px}
  .rail .panel{padding:16px 18px}
  .legendttl{font-size:12px;letter-spacing:.14em;text-transform:uppercase;color:var(--slate2);
    font-weight:600;margin:0 0 12px}
  .legrow{display:flex;align-items:center;gap:10px;padding:5px 6px;border-radius:8px;cursor:pointer;
    transition:background .12s}
  .legrow:hover,.legrow.on{background:var(--hair2)}
  .sw{width:14px;height:14px;border-radius:4px;border:1px solid rgba(22,32,46,.28);flex:none}
  .legrow .ln{font-size:13px;font-weight:550}
  .legrow .ld{font-size:11.5px;color:var(--slate2)}
  .keys{display:flex;gap:16px;margin-top:12px;padding-top:12px;border-top:1px solid var(--hair2);font-size:12px;color:var(--slate)}
  .keys span{display:inline-flex;align-items:center;gap:6px}
  .keybox{width:20px;height:12px;border-radius:3px;border:1.5px solid var(--ink);background:#fff}
  .keybox.ship{background:var(--ink)}
  .keybox.road{border-style:dashed}
  /* detail */
  #detail{min-height:120px}
  .d-eyebrow{font-size:11.5px;letter-spacing:.12em;text-transform:uppercase;color:var(--accent-2);font-weight:600}
  .d-title{font-family:var(--serif);font-size:20px;font-weight:600;line-height:1.15;margin:6px 0 2px;text-wrap:balance}
  .d-sub{font-size:12.5px;color:var(--slate2);font-variant-numeric:tabular-nums}
  .d-tags{display:flex;flex-wrap:wrap;gap:7px;margin:12px 0}
  .tag{font-size:11.5px;padding:3px 9px;border-radius:20px;border:1px solid var(--hair);color:var(--slate);white-space:nowrap}
  .tag.fam{color:var(--ink);font-weight:550}
  .tag.ship{background:var(--ink);color:#fff;border-color:var(--ink)}
  .tag.road{border-style:dashed}
  .tag.code{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;background:var(--accent-soft);color:var(--accent-2);border-color:transparent}
  .d-one{font-size:13.5px;color:var(--ink);margin:4px 0 0}
  .d-lab{font-size:11px;letter-spacing:.1em;text-transform:uppercase;color:var(--slate2);font-weight:600;margin:16px 0 4px}
  .d-innov{font-size:13px;color:var(--slate);line-height:1.5}
  .d-from{display:flex;flex-wrap:wrap;gap:6px;margin-top:6px}
  .d-from button{font:inherit;font-size:11.5px;cursor:pointer;background:var(--hair2);border:1px solid var(--hair);
    border-radius:7px;padding:3px 8px;color:var(--ink)}
  .d-from button:hover{border-color:var(--accent);color:var(--accent-2)}
  .d-rest{color:var(--slate2);font-size:12.5px}
  footer{margin-top:30px;padding-top:18px;border-top:1px solid var(--hair);color:var(--slate2);font-size:12.5px;max-width:80ch}
  footer a{color:var(--accent-2)}
  a.focusable:focus-visible,.node:focus-visible rect{outline:2px solid var(--accent);outline-offset:2px}
  @media(prefers-reduced-motion:reduce){*{transition:none!important}}
</style>

<div class="wrap">
  <header>
    <p class="eyebrow">TABenchmark &middot; model compendium</p>
    <h1>The evolution of traffic assignment</h1>
    <p class="thesis">Fifty years of models, drawn as the tree they actually are &mdash;
      each one answering a shortcoming of its parent. The trunk runs
      <b>principle &rarr; program &rarr; cost &rarr; solver</b>; every branch is a specific
      limitation someone refused to accept. Hover any model to trace the lineage that led to it.</p>
    <div class="chips" id="chips"></div>
  </header>

  <div class="work">
    <section class="panel">
      <div class="graphhead">
        <h2>Lineage graph</h2>
        <span class="hint">time flows downward &middot; hover to trace ancestry</span>
      </div>
      <div class="scroll" id="scroll"></div>
    </section>

    <aside class="rail">
      <div class="panel" id="detail"></div>
      <div class="panel">
        <p class="legendttl">Families</p>
        <div id="legend"></div>
        <div class="keys">
          <span><i class="keybox ship"></i> shipped</span>
          <span><i class="keybox road"></i> roadmap</span>
        </div>
      </div>
    </aside>
  </div>

  <footer id="foot"></footer>
</div>

<script>
const DATA = __DATA__;
const FAM = __FAM__;
const FAM_ORDER = __FAMORDER__;
const byKey = Object.fromEntries(DATA.map(d=>[d.k,d]));

// ---- layout: y by year, x by family lane (+ stagger) ----
const LANE_W=186, PAD_X=18, TOP=58, YSTEP=20;
const years=DATA.map(d=>d.year), Y0=Math.min(...years), Y1=Math.max(...years);
const yc=y=>TOP+(y-Y0)*YSTEP;
const W=PAD_X*2+LANE_W*FAM_ORDER.length, H=yc(Y1)+64;
const bucket={};
DATA.sort((a,b)=>a.year-b.year||a.k.localeCompare(b.k));
for(const d of DATA){
  const lane=FAM_ORDER.indexOf(d.fam), key=lane+"_"+d.year, n=bucket[key]||0; bucket[key]=n+1;
  d._x=PAD_X+lane*LANE_W+26+(n%2)*112;
  d._y=yc(d.year)+Math.floor(n/2)*17;
}
// ancestry closure
function ancestors(k,acc){acc=acc||new Set();for(const p of (byKey[k]?.from||[])){if(byKey[p]&&!acc.has(p)){acc.add(p);ancestors(p,acc);}}return acc;}

// ---- build SVG ----
const NS="http://www.w3.org/2000/svg";
let s=`<svg viewBox="0 0 ${W} ${H}" width="${W}" height="${H}" role="img" aria-label="Traffic assignment model lineage graph">`;
s+=`<rect width="${W}" height="${H}" fill="#fbfcfe"/>`;
FAM_ORDER.forEach((f,i)=>{const lx=PAD_X+i*LANE_W;
  s+=`<rect x="${lx}" y="0" width="${LANE_W}" height="${H}" fill="${FAM[f].c}" opacity="0.13"/>`;
  s+=`<text x="${lx+LANE_W/2}" y="22" text-anchor="middle" font-size="11.5" font-weight="700" fill="#2b3547" font-family="var(--sans)">${FAM[f].name}</text>`;
  s+=`<text x="${lx+LANE_W/2}" y="37" text-anchor="middle" font-size="10" fill="#828da2" font-family="var(--sans)">${FAM[f].desc}</text>`;});
for(let yr=Math.ceil(Y0/10)*10;yr<=Y1;yr+=10){const yy=yc(yr);
  s+=`<line x1="0" y1="${yy}" x2="${W}" y2="${yy}" stroke="#e6eaf1"/>`;
  s+=`<text x="5" y="${yy-3}" font-size="10" fill="#aab2c2" font-family="var(--sans)" font-variant-numeric="tabular-nums">${yr}</text>`;}
s+=`<defs><marker id="ar" markerWidth="7" markerHeight="7" refX="6" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6 Z" fill="#aeb6c6"/></marker>
<marker id="arh" markerWidth="8" markerHeight="8" refX="6.5" refY="3.2" orient="auto"><path d="M0,0 L6.5,3.2 L0,6.4 Z" fill="#0f6e78"/></marker></defs>`;
// edges
let edges="";
for(const d of DATA){for(const p of (d.from||[])){const par=byKey[p];if(!par)continue;
  const x1=par._x,y1=par._y+8,x2=d._x,y2=d._y-8,my=(y1+y2)/2;
  edges+=`<path class="edge" data-c="${d.k}" data-p="${p}" d="M${x1},${y1} C${x1},${my} ${x2},${my} ${x2},${y2}" fill="none" stroke="#c2c9d6" stroke-width="1" marker-end="url(#ar)" opacity="0.75"/>`;}}
s+=edges;
// nodes
for(const d of DATA){const dash=d.ship?"":`stroke-dasharray="3.5,2.5"`;
  s+=`<g class="node" data-k="${d.k}" tabindex="0" role="button" aria-label="${d.auth} ${d.year}">
    <rect x="${d._x-53}" y="${d._y-9}" width="106" height="18" rx="5" fill="${FAM[d.fam].c}" stroke="#2b3547" stroke-width="1" ${dash}/>
    <text x="${d._x}" y="${d._y+4}" text-anchor="middle" font-size="9.7" fill="#141d29">${d.label}</text></g>`;}
s+=`</svg>`;
document.getElementById("scroll").innerHTML=s;

// ---- chips ----
const nShip=DATA.filter(d=>d.ship).length;
document.getElementById("chips").innerHTML=[
  ["58","models mapped"],[String(nShip),"shipped & certified"],[String(58-nShip),"on the roadmap"],
  ["11","model families"],[`${Y0}–${Y1}`,"years of research"]
].map(c=>`<div class="chip"><b>${c[0]}</b><span>${c[1]}</span></div>`).join("");

// ---- legend ----
document.getElementById("legend").innerHTML=FAM_ORDER.map(f=>{
  const n=DATA.filter(d=>d.fam===f).length;
  return `<div class="legrow" data-fam="${f}"><span class="sw" style="background:${FAM[f].c}"></span>
    <span style="flex:1"><span class="ln">${FAM[f].name}</span> <span class="ld">&middot; ${n}</span><br><span class="ld">${FAM[f].desc}</span></span></div>`;}).join("");

// ---- footer ----
document.getElementById("foot").innerHTML=`Built from TABenchmark’s verified reference canon and the shipped solver code, grounded in the Boyles–Lownes–Unnikrishnan <em>Transportation Network Analysis</em> text. Nodes are models; edges point from a model to the descendants that built on it. Solid = shipped &amp; P1-certified today; dashed = queued on the roadmap. Companion prose: the model compendium (MODELS.md).`;

// ---- interaction ----
const svg=document.querySelector("svg");
const nodes=[...svg.querySelectorAll(".node")], edgeEls=[...svg.querySelectorAll(".edge")];
const detail=document.getElementById("detail");
let pinned=null;

function legendHTML(){return `<p class="d-eyebrow">how to read</p>
  <p class="d-title" style="font-size:17px">Hover a model</p>
  <p class="d-one">Move over any node to see what it did differently and light up the
  chain of ideas it descends from. Click to pin it; click empty space to release.
  Hover a family below to isolate that branch.</p>`;}

function famNodes(fam){return DATA.filter(d=>d.fam===fam).map(d=>d.k);}

function highlight(set,focusK){
  const inset=k=>set.has(k);
  nodes.forEach(g=>{const k=g.dataset.k;g.classList.toggle("dim",!inset(k));
    const r=g.querySelector("rect");r.setAttribute("stroke-width",k===focusK?"2.4":"1");
    r.setAttribute("stroke",k===focusK?"#0f6e78":"#2b3547");});
  edgeEls.forEach(e=>{const on=inset(e.dataset.c)&&inset(e.dataset.p);
    e.classList.toggle("dim",!on);
    e.setAttribute("stroke",on&&focusK?"#0f6e78":"#c2c9d6");
    e.setAttribute("stroke-width",on&&focusK?"1.7":"1");
    e.setAttribute("marker-end",on&&focusK?"url(#arh)":"url(#ar)");});
}
function clearHi(){nodes.forEach(g=>{g.classList.remove("dim");const r=g.querySelector("rect");
    r.setAttribute("stroke-width","1");r.setAttribute("stroke","#2b3547");});
  edgeEls.forEach(e=>{e.classList.remove("dim");e.setAttribute("stroke","#c2c9d6");
    e.setAttribute("stroke-width","1");e.setAttribute("marker-end","url(#ar)");});}

function showDetail(d){
  const anc=ancestors(d.k);
  const fromBtns=(d.from||[]).filter(p=>byKey[p]).map(p=>`<button data-go="${p}">${byKey[p].label}</button>`).join("");
  detail.innerHTML=`<p class="d-eyebrow">${FAM[d.fam].name}</p>
    <p class="d-title">${d.auth} <span style="color:var(--slate2);font-weight:500">${d.year}</span></p>
    <p class="d-sub">${d.title}</p>
    <div class="d-tags">
      <span class="tag fam" style="border-color:${FAM[d.fam].c};background:${FAM[d.fam].c}33">${FAM[d.fam].name}</span>
      <span class="tag ${d.ship?'ship':'road'}">${d.ship?'shipped':'roadmap'}</span>
      ${d.tab?`<span class="tag code">${d.tab}</span>`:''}
    </div>
    <p class="d-one">${d.one||''}</p>
    <p class="d-lab">What it did differently</p>
    <p class="d-innov">${d.innov||''}</p>
    ${fromBtns?`<p class="d-lab">Builds on</p><div class="d-from">${fromBtns}</div>`:
      `<p class="d-lab">Builds on</p><p class="d-rest">a root of the tree.</p>`}`;
  detail.querySelectorAll("[data-go]").forEach(b=>b.onclick=e=>{e.stopPropagation();focus(byKey[b.dataset.go],true);});
  const set=new Set([d.k,...anc]);highlight(set,d.k);
}
function focus(d,pin){if(pin){pinned=d.k;}showDetail(d);}
function reset(){pinned=null;clearHi();detail.innerHTML=legendHTML();}

nodes.forEach(g=>{const d=byKey[g.dataset.k];
  g.addEventListener("mouseenter",()=>{if(!pinned)showDetail(d);});
  g.addEventListener("focus",()=>{if(!pinned)showDetail(d);});
  g.addEventListener("click",e=>{e.stopPropagation();pinned===d.k?reset():focus(d,true);});
  g.addEventListener("keydown",e=>{if(e.key==="Enter"||e.key===" "){e.preventDefault();pinned===d.k?reset():focus(d,true);}});
});
svg.addEventListener("mouseleave",()=>{if(!pinned)reset();});
document.getElementById("scroll").addEventListener("click",e=>{if(e.target.tagName==="svg"||e.target.tagName==="rect"&&!e.target.closest(".node"))reset();});
// legend hover -> isolate family
document.querySelectorAll(".legrow").forEach(row=>{const fam=row.dataset.fam;
  row.addEventListener("mouseenter",()=>{if(pinned)return;row.classList.add("on");highlight(new Set(famNodes(fam)),null);});
  row.addEventListener("mouseleave",()=>{row.classList.remove("on");if(!pinned)clearHi();});
  row.addEventListener("click",()=>{const d=DATA.filter(x=>x.fam===fam).sort((a,b)=>a.year-b.year)[0];focus(d,true);});
});
reset();
</script>
"""

out = HTML.replace("__DATA__", DATA).replace("__FAM__", fam_js).replace("__FAMORDER__", fam_order)
(DOCS / "model-evolution.html").write_text(out)
print("wrote docs/model-evolution.html", len(out), "bytes")
