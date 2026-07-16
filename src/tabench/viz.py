"""House visualizer for TABenchmark networks, OD demand, and certified flows.

Optional module: matplotlib is an OPTIONAL extra (``pip install 'tabench[viz]'``),
kept out of the numpy/scipy core exactly like the torch / sumo / dtalite adapters.
The core stays dependency-free — this module is NEVER imported by ``tabench``'s
top-level ``__init__``, so ``import tabench`` works without matplotlib; only
``import tabench.viz`` (or calling one of its functions) needs it, and a missing
matplotlib is reported with the install hint rather than a bare ImportError.

Public API
----------
* :func:`node_positions` — deterministic node layout (explicit > cached TNTP
  coordinates > hand layout > layered BFS fallback).
* :func:`plot_network_flows` — a directed network coloured/sized by link flow.
* :func:`plot_od_demand` — an OD-matrix heatmap.
* :func:`plot_flow_scatter` — model-vs-reference link-flow scatter (the P1 story).
* :func:`compare_models` — a panel grid of models against a reference, plus the
  scatter, on one shared colour scale.

Everything is deterministic: identical inputs give identical positions and an
identical figure structure (no randomness anywhere). House style is applied
through an ``rc_context`` at figure creation plus explicit per-artist colours;
nothing mutates the global matplotlib rcParams, so a user's own session is left
untouched.
"""

from __future__ import annotations

import math
from collections import deque

import numpy as np

from .core.scenario import Network

# House-pattern guarded import (mirrors tabench.models.__init__): swallow ONLY a
# genuinely-missing matplotlib and defer the error to call time with an install
# hint; any other ImportError is a real bug in matplotlib and must propagate.
try:
    import matplotlib
    import matplotlib.colors as mcolors
    import matplotlib.patches as mpatches
    from matplotlib.cm import ScalarMappable
    from matplotlib.figure import Figure

    _HAS_MPL = True
except ModuleNotFoundError as exc:  # pragma: no cover - exercised by matplotlib-free installs
    if exc.name != "matplotlib":
        raise
    _HAS_MPL = False

__all__ = [
    "node_positions",
    "plot_network_flows",
    "plot_od_demand",
    "plot_flow_scatter",
    "compare_models",
]

# --- house style ------------------------------------------------------------
_SURFACE = "#fcfcfb"  # light surface
_TEXT = "#0b0b0b"  # primary text
_SECONDARY = "#52514e"  # secondary text / recessive spines
_GRID = "#e5e4e0"  # grid + guide lines
_PRIMARY = "#2a78d6"  # first series / accent
_GT_TINT = "#f2f7fd"  # faint tint marking the reference (ground-truth) panel
# Series order: the two house colours first, then a tasteful extension so 6+
# models stay distinguishable; cycled by _series_color for anything larger.
_SERIES = ("#2a78d6", "#1baf7a", "#e8833a", "#9b59b6", "#d64550", "#d9a527", "#4a4a48")
# Distinct marker shapes so coincident scatter series (e.g. fw/cfw/bfw at the same UE
# flow) stay separable rather than the last one hiding the rest.
_MARKERS = ("o", "s", "^", "D", "v", "P", "X")
_WIDTH_MIN, _WIDTH_MAX = 1.2, 6.0  # link-width range (points)

# Figure-level defaults applied via rc_context (never a global mutation): a leak
# into the user's session is a documented program hazard.
_HOUSE_RC = {
    "figure.facecolor": _SURFACE,
    "savefig.facecolor": _SURFACE,
    "axes.facecolor": _SURFACE,
    "text.color": _TEXT,
    "axes.labelcolor": _TEXT,
    "axes.titlecolor": _TEXT,
    "axes.edgecolor": _SECONDARY,
    "xtick.color": _SECONDARY,
    "ytick.color": _SECONDARY,
    "grid.color": _GRID,
    "legend.frameon": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 9,
}

# Auto-annotate per-link flow counts at or below this link count (dense networks
# would overprint; the built-ins and Sioux Falls stay legible below it).
_ANNOTATE_MAX_LINKS = 40

# Framing / arrow-trim tuning, all as fractions of the TRUE data span (see _span):
_MARGIN_FRAC = 0.10  # axis padding each side (R2: tight framing, was a loose 0.18)
_INSET_PROP = 0.16  # per-end arrow trim as a fraction of link length (short links; attack8)
# R1 absolute cap: a long link's per-end trim is capped at _INSET_CAP_C * (node marker
# radius in data units), so long links reach their nodes instead of stopping 16% short
# (outlying Sioux Falls nodes read as disconnected) while short links keep the proportional
# trim above. c is calibrated EMPIRICALLY, not to the "small c ~ 1.5-2.5" first guess: in the
# house framing the marker is small enough that Braess's longest link (the 3->4 bypass) has an
# uncapped trim of ~3.57 marker-radii, so any c below that would shorten Braess's arrows and
# change "the render that is right". c = 4.0 sits just above it — the built-ins keep their
# proportional trim byte-identical, and only genuinely long links (Sioux Falls's outlying-node
# links, ~6.9 marker-radii uncapped) are pulled in to ~4.9.
_INSET_CAP_C = 4.0
_DRAW_LONG_IN = 4.8  # drawing region's longer side (inches) at fig-scale 1 (Braess-sized)


def _require_matplotlib() -> None:
    if not _HAS_MPL:
        raise ModuleNotFoundError(
            "tabench.viz needs matplotlib; install it with `pip install 'tabench[viz]'`",
            name="matplotlib",
        )


def _series_color(i: int) -> str:
    """The i-th series colour, cycling the house palette for 7+ series."""
    return _SERIES[i % len(_SERIES)]


def _flow_cmap():
    """Single-hue blue sequential for flow magnitude (on-brand, light->deep)."""
    return mcolors.LinearSegmentedColormap.from_list(
        "tabench_flow", ["#cfe0f5", _PRIMARY, "#0b3d78"]
    )


def _node_scale(n_nodes: int) -> float:
    """Node marker / font scale: 1.0 for small nets (Braess), shrinking for dense
    ones so a 24-node Sioux Falls or a 100-node fallback stays legible (M1/M13)."""
    return min(max(8.0 / max(n_nodes, 1), 0.18), 1.0)


def _fig_scale(n_nodes: int) -> float:
    """Figure-size scale: grows the canvas with node count so real-coordinate links
    (short in display space) draw as visible arrows instead of being erased (M1)."""
    return min(max((n_nodes / 6.0) ** 0.5, 1.0), 2.2)


# --- node layout ------------------------------------------------------------
# Hand layouts for the built-ins whose topology has a canonical drawing.
# Convention: origin zone(s) west (small x), destination zone(s) east (large x).
# Braess is the classic diamond: 1 = origin (west), 2 = destination (east),
# 3 = north and 4 = south intersections, with the 3->4 bypass drawn vertically
# down the middle. (Verified against tabench.data.builtin.braess_scenario: nodes
# 1,2 are the zones, links 1->3, 1->4, 3->4, 3->2, 4->2.)
_HAND_LAYOUTS: dict[str, dict[int, tuple[float, float]]] = {
    "braess": {1: (0.0, 0.0), 2: (2.0, 0.0), 3: (1.0, 1.0), 4: (1.0, -1.0)},
}


def _parse_node_coords(path) -> dict[int, tuple[float, float]]:
    """Defensively parse a TNTP ``*_node.tntp`` coordinate file.

    Format: an optional ``Node X Y`` header then whitespace rows ``id x y ;``.
    Malformed or header lines are skipped, never fatal (this is plotting only).
    """
    positions: dict[int, tuple[float, float]] = {}
    for raw in path.read_text().splitlines():
        line = raw.strip().rstrip(";").strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 3:
            continue
        try:
            node = int(parts[0])
            x = float(parts[1])
            y = float(parts[2])
        except ValueError:
            continue  # header row or non-numeric line
        if not (math.isfinite(x) and math.isfinite(y)):
            continue  # NaN/inf coords would crash far away at set_xlim — drop the row
        positions[node] = (x, y)
    return positions


def _coords_from_cache(network: Network) -> dict[int, tuple[float, float]] | None:
    """Cached TNTP node coordinates for ``network.name``, or None.

    NEVER downloads: it reads only an already-cached ``node`` file for a network
    whose registry key equals ``network.name`` (Sioux Falls has one). Any
    registry/cache/parse hiccup degrades to None so plotting falls through to a
    layout that needs no data.
    """
    try:
        from .data.fetcher import cache_dir
        from .data.registry import REGISTRY
    except ImportError:  # pragma: no cover - data layer is always present
        return None
    spec = REGISTRY.get(network.name)
    if spec is None or "node" not in spec.files:
        return None
    filename = spec.files["node"][0]
    path = cache_dir() / spec.key / filename
    if not path.exists():
        return None
    try:
        return _parse_node_coords(path)
    except (OSError, UnicodeDecodeError):
        # A corrupt or non-UTF-8 cached file must degrade to None (fall back to a
        # data-free layout), never crash — this is plotting only.
        return None


def _layered_positions(network: Network) -> dict[int, tuple[float, float]]:
    """Deterministic layered fallback: x = hop distance from the zone nodes.

    Multi-source (undirected) BFS from the zone nodes (1..n_zones) gives each
    node a layer; nodes unreachable from any zone are pushed one layer past the
    deepest. Within a layer nodes spread vertically, centred, ordered by id — so
    two calls are byte-identical and no two nodes coincide within a layer.
    """
    n = network.n_nodes
    adj: dict[int, set[int]] = {i: set() for i in range(1, n + 1)}
    for a, b in zip(network.init_node.tolist(), network.term_node.tolist(), strict=True):
        adj[int(a)].add(int(b))
        adj[int(b)].add(int(a))

    dist: dict[int, int] = {}
    queue: deque[int] = deque()
    for source in range(1, network.n_zones + 1):
        dist[source] = 0
        queue.append(source)
    while queue:
        u = queue.popleft()
        for v in sorted(adj[u]):
            if v not in dist:
                dist[v] = dist[u] + 1
                queue.append(v)
    unreached_layer = (max(dist.values()) + 1) if dist else 0
    for i in range(1, n + 1):
        dist.setdefault(i, unreached_layer)

    layers: dict[int, list[int]] = {}
    for node in range(1, n + 1):
        layers.setdefault(dist[node], []).append(node)
    positions: dict[int, tuple[float, float]] = {}
    for layer, nodes in layers.items():
        k = len(nodes)
        for idx, node in enumerate(sorted(nodes)):
            positions[node] = (float(layer), float(idx) - (k - 1) / 2.0)
    return positions


def node_positions(
    network: Network, pos: dict[int, tuple[float, float]] | None = None
) -> dict[int, tuple[float, float]]:
    """Resolve 1-based node positions for ``network``.

    Resolution order (first that yields every node wins): the explicit ``pos``
    argument, cached TNTP node coordinates for this network, a hand layout for
    the built-ins, then the deterministic layered BFS fallback. The result is a
    plain ``{node_id: (x, y)}`` dict covering nodes ``1..n_nodes``.
    """
    all_nodes = range(1, network.n_nodes + 1)
    if pos is not None:
        pos = {int(k): v for k, v in pos.items()}
        missing = [i for i in all_nodes if i not in pos]
        if missing:
            raise ValueError(
                f"explicit pos is missing positions for node ids {missing} "
                f"(network '{network.name}' has nodes 1..{network.n_nodes})"
            )
        return {i: (float(pos[i][0]), float(pos[i][1])) for i in all_nodes}
    cached = _coords_from_cache(network)
    # Require EXACTLY the network's node set: a superset (e.g. a 24-node Sioux Falls
    # coordinate file cached under a 4-node network of the same name) must NOT silently
    # place the small network at the wrong coordinates — fall through instead.
    exact = cached is not None and len(cached) == network.n_nodes
    if exact and all(i in cached for i in all_nodes):
        return {i: cached[i] for i in all_nodes}
    hand = _HAND_LAYOUTS.get(network.name)
    if hand is not None and all(i in hand for i in all_nodes):
        return {i: hand[i] for i in all_nodes}
    return _layered_positions(network)


# --- drawing primitives -----------------------------------------------------
def _style_axes(ax) -> None:
    """Recessive house styling for a data axes (scatter, heatmap frame)."""
    ax.set_facecolor(_SURFACE)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(_SECONDARY)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=_SECONDARY, labelsize=8)
    ax.set_axisbelow(True)


def _span(positions: dict[int, tuple[float, float]]) -> float:
    """True data extent (larger of the x/y span). Floored to 1.0 ONLY when every node
    coincides (a degenerate layout), NOT unconditionally: on small-coordinate networks
    (Sioux Falls WGS84 lon/lat, x-extent ~0.06) an unconditional 1.0 floor inflated every
    span-relative quantity — margin, reverse-link offset, label offset — by ~9x, cramming
    the network into the canvas centre and pushing the reverse-link offset past the link
    length so arrows pointed sideways and outlying nodes read as disconnected (R1/R2)."""
    xs = [p[0] for p in positions.values()]
    ys = [p[1] for p in positions.values()]
    extent = max(max(xs) - min(xs), max(ys) - min(ys))
    return extent if extent > 0.0 else 1.0


def _figsize(
    x_range: float, y_range: float, n_nodes: int, base: float = _DRAW_LONG_IN
) -> tuple[float, float]:
    """Figure (or panel) size matching the DATA aspect ratio (R2): with ``aspect='equal'``
    a figure whose shape differs from the data's leaves the network in a tall/wide band of
    empty canvas. The longer side is ``base * _fig_scale`` (Braess-sized at scale 1); the
    shorter side follows the data box, so the axes fills the frame at every aspect."""
    long_in = base * _fig_scale(n_nodes)
    aspect = y_range / x_range if x_range > 0 else 1.0  # height / width of the data box
    if aspect >= 1.0:  # taller than wide
        return long_in / aspect, long_in
    return long_in, long_in * aspect


def _ranges(positions: dict[int, tuple[float, float]]) -> tuple[float, float]:
    """The (x, y) axis ranges the network will occupy, margin included — the data box the
    figure is sized to. Kept consistent with the xlim/ylim set inside ``_draw_network``."""
    xs = [p[0] for p in positions.values()]
    ys = [p[1] for p in positions.values()]
    margin = _MARGIN_FRAC * _span(positions)
    return (max(xs) - min(xs)) + 2 * margin, (max(ys) - min(ys)) + 2 * margin


def _draw_network(
    ax,
    network: Network,
    flows: np.ndarray,
    positions: dict[int, tuple[float, float]],
    annotate: bool | None,
    vmin: float | None,
    vmax: float | None,
    cmap,
    draw_long_in: float | None = None,
):
    """Draw directed links (width + colour by flow) and nodes onto ``ax``.

    ``draw_long_in`` is the drawing region's longer side in inches; the caller passes it
    (it sized the figure) so the R1 arrow-trim cap can express the node-marker radius in
    data units. With ``None`` (a user-supplied ``ax``) it is estimated from the figure size.

    Returns ``(norm, cmap)`` so a caller can attach a matching colorbar.
    """
    flows = np.asarray(flows, dtype=float)
    if annotate is None:
        annotate = network.n_links <= _ANNOTATE_MAX_LINKS
    if cmap is None:
        cmap = _flow_cmap()
    finite = flows[np.isfinite(flows)]
    if vmin is None:
        vmin = float(finite.min()) if finite.size else 0.0
    if vmax is None:
        vmax = float(finite.max()) if finite.size else 1.0
    if vmax <= vmin:
        vmax = vmin + 1.0
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    ax.set_facecolor(_SURFACE)
    ax.set_aspect("equal")
    ax.axis("off")

    # Node markers / fonts / arrow widths scale down for dense networks so Sioux Falls
    # (24 nodes) or a large layered fallback stays legible while Braess is unchanged.
    nscale = _node_scale(network.n_nodes)
    zone_s = max(440.0 * nscale, 60.0)
    thru_s = max(360.0 * nscale, 45.0)
    label_fs = max(7.0 * nscale**0.5, 5.0)
    id_fs = max(8.0 * nscale**0.5, 4.5)

    span = _span(positions)
    margin = _MARGIN_FRAC * span
    pair_offset = 0.035 * span  # separate reverse links so they don't overprint
    label_offset = 0.06 * span
    link_set = {
        (int(a), int(b))
        for a, b in zip(network.init_node.tolist(), network.term_node.tolist(), strict=True)
    }

    # R1 arrow-trim cap: express the node-marker radius in DATA units, then cap each arrow's
    # per-end trim at _INSET_CAP_C marker-radii. The zone marker has area zone_s points^2
    # (radius ~ 0.5*sqrt(area) points); data-units-per-inch = data_range / drawing_inches on
    # the binding (longer) axis, with aspect='equal' the same on both. Short links keep the
    # proportional 16% trim (the attack8 fix); long links (small marker relative to span, e.g.
    # outlying Sioux Falls nodes) are pulled in so their arrows reach the node markers.
    xs = [p[0] for p in positions.values()]
    ys = [p[1] for p in positions.values()]
    x_range = (max(xs) - min(xs)) + 2 * margin
    y_range = (max(ys) - min(ys)) + 2 * margin
    range_longer = max(x_range, y_range, 1e-12)
    if draw_long_in is None:  # user-supplied ax: estimate the axes size from the figure
        draw_long_in = 0.85 * float(max(ax.figure.get_size_inches()))
    inches_per_data = draw_long_in / range_longer
    marker_r_pt = 0.5 * math.sqrt(zone_s)  # zone-marker radius in points
    marker_r_data = (marker_r_pt / 72.0) / inches_per_data
    inset_cap = _INSET_CAP_C * marker_r_data  # absolute per-end trim ceiling, data units

    # Trim each arrow in DATA space. The M1 lesson: a fixed point-based shrink erases every
    # link shorter than the shrink budget on real WGS84 coordinates (Sioux Falls dropped
    # 70/76 links); a data-space trim always leaves a visible, correctly-directed arrow.
    for k in range(network.n_links):
        a = int(network.init_node[k])
        b = int(network.term_node[k])
        xa, ya = positions[a]
        xb, yb = positions[b]
        dx, dy = xb - xa, yb - ya
        length = math.hypot(dx, dy)
        if length == 0.0:
            continue  # coincident endpoints: nothing to draw
        # unit perpendicular; offset only when the reverse link also exists
        px, py = -dy / length, dx / length
        off = pair_offset if (b, a) in link_set else 0.0
        xa2, ya2 = xa + px * off, ya + py * off
        xb2, yb2 = xb + px * off, yb + py * off
        # per-end trim: proportional for short links, capped near the marker for long ones
        frac = min(_INSET_PROP * length, inset_cap) / length
        sx, sy = xa2 + frac * (xb2 - xa2), ya2 + frac * (yb2 - ya2)
        ex, ey = xb2 - frac * (xb2 - xa2), yb2 - frac * (yb2 - ya2)

        f = float(flows[k])
        if math.isfinite(f):
            t = min(max((f - vmin) / (vmax - vmin), 0.0), 1.0)
            color = cmap(t)
        else:
            t = 0.0
            color = _SECONDARY
        width = (_WIDTH_MIN + (_WIDTH_MAX - _WIDTH_MIN) * t) * (0.5 + 0.5 * nscale)
        ax.annotate(
            "",
            xy=(ex, ey),
            xytext=(sx, sy),
            arrowprops=dict(
                arrowstyle="-|>",
                color=color,
                lw=width,
                shrinkA=0,
                shrinkB=0,
                mutation_scale=(10 + 1.5 * width) * (0.4 + 0.6 * nscale),
            ),
            zorder=2,
        )
        if annotate:
            label = "nan" if not math.isfinite(f) else f"{f:.3g}"
            ax.text(
                (sx + ex) / 2 + px * label_offset,
                (sy + ey) / 2 + py * label_offset,
                label,
                ha="center",
                va="center",
                fontsize=label_fs,
                color=_TEXT,
                zorder=3,
            )

    # Node fills sit BELOW the arrows (zorder 1.5 < 2): when two nodes are closer than a
    # marker diameter (the attack8 pathology), the link's arrow still draws visibly OVER
    # the markers instead of being occluded by them; node-id labels stay on top (zorder 4).
    for node, (x, y) in positions.items():
        is_zone = node <= network.n_zones
        ax.scatter(
            [x],
            [y],
            s=zone_s if is_zone else thru_s,
            marker="s" if is_zone else "o",
            facecolor=_GT_TINT if is_zone else "#ffffff",
            edgecolor=_PRIMARY if is_zone else _SECONDARY,
            linewidths=(1.4 if is_zone else 1.1) * nscale**0.5,
            zorder=1.5,
        )
        ax.text(
            x, y, str(node), ha="center", va="center", fontsize=id_fs, color=_TEXT, zorder=4
        )

    ax.set_xlim(min(xs) - margin, max(xs) + margin)
    ax.set_ylim(min(ys) - margin, max(ys) + margin)
    return norm, cmap


def _draw_od(ax, matrix: np.ndarray):
    """Draw an OD-matrix heatmap onto ``ax`` (annotated for small zone counts)."""
    matrix = np.asarray(matrix, dtype=float)
    n = matrix.shape[0]
    cmap = _flow_cmap()
    # Finite-masked max: a single NaN cell must not collapse the whole colour scale
    # (vmax=nan slips a `<= 0` guard, and every distinct demand then renders identically).
    finite = matrix[np.isfinite(matrix)]
    vmax = float(finite.max()) if finite.size else 1.0
    if not (vmax > 0.0):
        vmax = 1.0
    im = ax.imshow(matrix, cmap=cmap, origin="upper", vmin=0.0, vmax=vmax, aspect="equal")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(range(1, n + 1), fontsize=8)
    ax.set_yticklabels(range(1, n + 1), fontsize=8)
    ax.set_xlabel("destination zone")
    ax.set_ylabel("origin zone")
    ax.tick_params(colors=_SECONDARY, length=0)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    if n <= 15:
        for i in range(n):
            for j in range(n):
                v = float(matrix[i, j])
                ax.text(
                    j,
                    i,
                    f"{v:.3g}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="#ffffff" if v / vmax > 0.55 else _TEXT,
                )
    return im


def _draw_scatter(ax, ref_flows: np.ndarray, model_flows: dict[str, np.ndarray], ref_label: str):
    """Draw a model-vs-reference link-flow scatter (one series per model)."""
    _style_axes(ax)
    ref_flows = np.asarray(ref_flows, dtype=float)
    ax.grid(True, color=_GRID, linewidth=0.6)

    finite_ref = ref_flows[np.isfinite(ref_flows)]
    hi = float(finite_ref.max(initial=0.0))
    # Fold the reference minimum into lo (initial=0 keeps the origin for the usual
    # nonnegative case): a negative reference flow must not plot off-canvas (m6).
    lo = float(finite_ref.min(initial=0.0))
    for i, (name, flows) in enumerate(model_flows.items()):
        f = np.asarray(flows, dtype=float)
        mask = np.isfinite(f) & np.isfinite(ref_flows)
        if mask.any():
            hi = max(hi, float(f[mask].max()))
            lo = min(lo, float(f[mask].min()))
        # Hollow markers, shape-cycled and size-stepped by series index, so coincident
        # series (fw/cfw/bfw at the same UE flow) show as nested rings, deterministically.
        ax.scatter(
            ref_flows[mask],
            f[mask],
            s=max(64 - 8 * i, 20),
            marker=_MARKERS[i % len(_MARKERS)],
            facecolor="none",
            edgecolors=_series_color(i),
            linewidths=1.4,
            label=name,
            alpha=0.9,
            zorder=3 + i,
        )
    pad = 0.05 * (hi - lo or 1.0)
    ax.plot(
        [lo - pad, hi + pad],
        [lo - pad, hi + pad],
        color=_GRID,
        linewidth=1.0,
        linestyle="--",
        zorder=1,
    )
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_aspect("equal")
    ax.set_xlabel(f"{ref_label} link flow")
    ax.set_ylabel("model link flow")
    ax.set_title("model vs reference", fontsize=9)
    ax.legend(fontsize=7, frameon=False)


# --- public plotting API ----------------------------------------------------
def plot_network_flows(
    network: Network,
    flows: np.ndarray,
    ax=None,
    pos: dict[int, tuple[float, float]] | None = None,
    annotate: bool | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    cmap=None,
):
    """Draw ``network`` with each directed link sized and coloured by its flow.

    ``annotate`` prints per-link flow counts (auto-on for <= 40 links); reverse
    link pairs are perpendicular-offset so they never overprint. ``vmin``/``vmax``
    fix a shared colour scale (compare_models passes them so panels are
    comparable). With ``ax=None`` a new figure (with a colorbar) is created and
    returned; otherwise the flows are drawn on ``ax`` and its figure returned.
    """
    _require_matplotlib()
    flows = np.asarray(flows, dtype=float)
    if flows.shape != (network.n_links,):
        raise ValueError(f"flows must have shape ({network.n_links},), got {flows.shape}")
    positions = node_positions(network, pos)
    if ax is not None:
        _draw_network(ax, network, flows, positions, annotate, vmin, vmax, cmap)
        return ax.figure
    with matplotlib.rc_context(_HOUSE_RC):
        x_range, y_range = _ranges(positions)
        fig_w, fig_h = _figsize(x_range, y_range, network.n_nodes)
        fig = Figure(figsize=(fig_w, fig_h), layout="constrained")
        ax = fig.subplots()
        draw_long = 0.85 * max(fig_w, fig_h)  # drawing region after the colorbar/title
        norm, used_cmap = _draw_network(
            ax, network, flows, positions, annotate, vmin, vmax, cmap, draw_long_in=draw_long
        )
        sm = ScalarMappable(norm=norm, cmap=used_cmap)
        sm.set_array([])
        fig.colorbar(sm, ax=ax, fraction=0.045, pad=0.02, label="link flow")
        ax.set_title(f"{network.name}: link flows", fontsize=10)
    return fig


def plot_od_demand(demand, ax=None):
    """Heatmap of an OD demand matrix (annotated cells for small zone counts).

    Accepts a :class:`~tabench.core.scenario.Demand` (or any object with a
    ``matrix`` attribute) or a raw square array. A degenerate single-OD matrix
    still renders as annotated cells rather than a flat colorbar.
    """
    _require_matplotlib()
    matrix = np.asarray(getattr(demand, "matrix", demand), dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"OD demand must be a square matrix, got shape {matrix.shape}")
    if ax is not None:
        _draw_od(ax, matrix)
        return ax.figure
    with matplotlib.rc_context(_HOUSE_RC):
        fig = Figure(figsize=(4.4, 3.8), layout="constrained")
        ax = fig.subplots()
        im = _draw_od(ax, matrix)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02, label="OD demand")
        ax.set_title("OD demand", fontsize=10)
    return fig


def plot_flow_scatter(
    reference: tuple[str, np.ndarray], model_flows: dict[str, np.ndarray], ax=None
):
    """Scatter each model's link flows against a ``(label, flows)`` reference.

    Points off the ``y = x`` guide are model-vs-reference disagreements — the
    visual form of the P1 story (a censored surrogate lands visibly off-diagonal).
    """
    _require_matplotlib()
    ref_label, ref_flows = str(reference[0]), np.asarray(reference[1], dtype=float)
    # Validate every array up front (m4): a long array silently pollutes the scale, a
    # short one dies mid-render leaking a half-drawn figure.
    if ref_flows.ndim != 1:
        raise ValueError(f"reference flows must be 1-D, got shape {ref_flows.shape}")
    for name, f in model_flows.items():
        if np.asarray(f).shape != ref_flows.shape:
            raise ValueError(
                f"model '{name}' flows must have shape {ref_flows.shape} to match the "
                f"reference, got {np.asarray(f).shape}"
            )
    if ax is not None:
        _draw_scatter(ax, ref_flows, model_flows, ref_label)
        return ax.figure
    with matplotlib.rc_context(_HOUSE_RC):
        fig = Figure(figsize=(4.8, 4.6), layout="constrained")
        ax = fig.subplots()
        _draw_scatter(ax, ref_flows, model_flows, ref_label)
    return fig


def compare_models(
    scenario,
    model_flows: dict[str, np.ndarray],
    reference: tuple[str, np.ndarray] | None = None,
    pos: dict[int, tuple[float, float]] | None = None,
):
    """Panel grid comparing each model's link flows on one shared colour scale.

    One network panel per model, plus (when ``reference`` is given) a
    distinguished ``GT: <label>`` panel FIRST and a final model-vs-reference
    scatter panel. All network panels share a single flow colour scale and one
    colorbar. Returns the :class:`~matplotlib.figure.Figure`; the caller saves it.
    """
    _require_matplotlib()
    network = scenario.network
    if not model_flows and reference is None:
        raise ValueError(
            "compare_models has nothing to plot: model_flows is empty and no reference given"
        )
    n_links = network.n_links

    def _check(label: str, arr) -> np.ndarray:
        arr = np.asarray(arr, dtype=float)
        if arr.shape != (n_links,):  # m4: reject wrong-length flows up front
            raise ValueError(
                f"{label} flows must have shape ({n_links},), got {arr.shape}"
            )
        return arr

    positions = node_positions(network, pos)

    panels: list[tuple[str, np.ndarray, bool]] = []
    ref_flows = None
    ref_label = None
    if reference is not None:
        ref_label = str(reference[0])
        ref_flows = _check(f"reference '{ref_label}'", reference[1])
        panels.append((f"GT: {ref_label}", ref_flows, True))
    for name in model_flows:
        panels.append((str(name), _check(f"model '{name}'", model_flows[name]), False))

    stacked = np.concatenate([f.ravel() for _, f, _ in panels]) if panels else np.zeros(1)
    finite = stacked[np.isfinite(stacked)]
    vmin = float(finite.min()) if finite.size else 0.0
    vmax = float(finite.max()) if finite.size else 1.0
    if vmax <= vmin:
        vmax = vmin + 1.0
    cmap = _flow_cmap()
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    has_scatter = reference is not None
    total = len(panels) + (1 if has_scatter else 0)
    ncols = 1 if total <= 1 else (2 if total <= 4 else 3)
    nrows = math.ceil(total / ncols)

    x_range, y_range = _ranges(positions)
    panel_w, panel_h = _figsize(x_range, y_range, network.n_nodes, base=3.5)
    draw_long = 0.85 * max(panel_w, panel_h)  # per-panel drawing region
    with matplotlib.rc_context(_HOUSE_RC):
        fig = Figure(
            figsize=(panel_w * ncols, panel_h * nrows), layout="constrained"
        )
        axgrid = fig.subplots(nrows, ncols, squeeze=False)
        flat = axgrid.ravel().tolist()
        net_axes = []
        for ax, (title, flows, is_ref) in zip(flat, panels, strict=False):
            _draw_network(
                ax, network, flows, positions, None, vmin, vmax, cmap, draw_long_in=draw_long
            )
            if is_ref:
                # The panel patch is hidden by axis("off"), so mark the reference panel
                # with an explicit tinted, blue-bordered background rectangle (m8: the
                # set_facecolor line it replaces was dead code).
                ax.add_patch(
                    mpatches.Rectangle(
                        (0, 0), 1, 1, transform=ax.transAxes, facecolor=_GT_TINT,
                        edgecolor=_PRIMARY, linewidth=1.6, zorder=-1, clip_on=False,
                    )
                )
                ax.set_title(title, color=_PRIMARY, fontweight="bold", fontsize=9)
            else:
                ax.set_title(title, color=_TEXT, fontsize=9)
            net_axes.append(ax)
        used = len(panels)
        if has_scatter:
            _draw_scatter(flat[used], ref_flows, model_flows, ref_label)
            used += 1
        for ax in flat[used:]:  # drop any unused grid cell so fig.axes is exact
            fig.delaxes(ax)
        sm = ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        fig.colorbar(sm, ax=net_axes, fraction=0.025, pad=0.01, label="link flow")
        fig.suptitle(f"{scenario.name}: link flows by model", color=_TEXT, fontsize=11)
    return fig
