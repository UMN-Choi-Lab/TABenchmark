"""Tests for the house visualizer (``tabench.viz``) and ``demo_quickstart --viz``.

matplotlib is an OPTIONAL extra, so this whole module is skipped on a
matplotlib-less install (``pytest.importorskip('matplotlib')``, mirroring the
torch/sumo/dtalite gating). CI installs ``.[dev,viz]`` on both core matrix legs,
so these run on 3.10 and 3.12. Agg is forced immediately after the import so the
suite never needs a display.
"""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from conftest import load_or_skip  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from tabench import braess_scenario, two_route_scenario, viz  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEMO = _REPO_ROOT / "demos" / "demo_quickstart.py"


def _fake_flows(n_links: int, spread: float, seed: int) -> np.ndarray:
    """Deterministic nonnegative pseudo-flows for a smoke figure (no solver)."""
    rng = np.random.default_rng(seed)
    return np.abs(rng.normal(3.0, spread, n_links))


# --- node layout ------------------------------------------------------------
def test_node_positions_deterministic_and_braess_hand_layout():
    net = braess_scenario().network
    a = viz.node_positions(net)
    b = viz.node_positions(net)
    assert a == b  # byte-identical across calls (determinism is a measured claim)
    assert set(a) == {1, 2, 3, 4}
    # Documented convention: origin (zone 1) west of destination (zone 2).
    assert a[1][0] < a[2][0]
    # Braess diamond: node 3 north (above) node 4 south (below).
    assert a[3][1] > a[4][1]


def test_layered_fallback_on_scenario_without_coords_or_hand_layout():
    # two-route: network.name is "two-route" (not "braess", not a registry key),
    # so neither cached coords nor a hand layout apply -> layered BFS fallback.
    net = two_route_scenario().network
    assert net.name not in viz._HAND_LAYOUTS
    pos = viz.node_positions(net)
    assert viz.node_positions(net) == pos  # deterministic
    assert set(pos) == {1, 2, 3, 4}
    # No two nodes coincide.
    assert len({(round(x, 9), round(y, 9)) for x, y in pos.values()}) == 4
    # Zone nodes 1,2 are layer 0; through-nodes 3,4 are layer 1 (x = hop distance).
    assert pos[1][0] == pos[2][0] == 0.0
    assert pos[3][0] == pos[4][0] == 1.0


def test_explicit_pos_overrides_everything():
    net = braess_scenario().network
    explicit = {1: (5.0, 5.0), 2: (6.0, 6.0), 3: (7.0, 7.0), 4: (8.0, 8.0)}
    assert viz.node_positions(net, pos=explicit) == explicit


# --- figures ----------------------------------------------------------------
def test_compare_models_smoke_panel_count_and_savefig(tmp_path):
    scenario = braess_scenario()
    n = scenario.network.n_links
    model_flows = {
        "aon": _fake_flows(n, 0.4, 1),
        "fw": _fake_flows(n, 0.1, 2),
        "toy": _fake_flows(n, 1.5, 3),
    }
    reference = ("bfw", np.array([4.0, 2.0, 2.0, 2.0, 4.0]))
    fig = viz.compare_models(scenario, model_flows, reference=reference)
    assert isinstance(fig, Figure)
    # 1 GT panel + 3 model panels + 1 scatter panel + 1 shared colorbar = 6 axes.
    assert len(fig.axes) == len(model_flows) + 3
    out = tmp_path / "compare.png"
    fig.savefig(out, dpi=100)
    plt.close(fig)
    assert out.exists() and out.stat().st_size > 0


def test_compare_models_without_reference_has_no_scatter():
    scenario = braess_scenario()
    n = scenario.network.n_links
    fig = viz.compare_models(scenario, {"a": _fake_flows(n, 0.4, 1), "b": _fake_flows(n, 0.4, 2)})
    # 2 model panels + colorbar, no GT panel and no scatter.
    assert len(fig.axes) == 3
    plt.close(fig)


def test_plot_network_flows_returns_figure(tmp_path):
    scenario = braess_scenario()
    flows = np.array([4.0, 2.0, 2.0, 2.0, 4.0])
    fig = viz.plot_network_flows(scenario.network, flows)
    assert isinstance(fig, Figure)
    out = tmp_path / "flows.png"
    fig.savefig(out, dpi=100)
    plt.close(fig)
    assert out.stat().st_size > 0


def test_plot_network_flows_rejects_wrong_shape():
    scenario = braess_scenario()
    with pytest.raises(ValueError, match="shape"):
        viz.plot_network_flows(scenario.network, np.zeros(3))


def test_od_heatmap_on_tiny_braess_od(tmp_path):
    # braess is essentially a 1-OD scenario (only [1->2] positive): it must still
    # render annotated cells, not a degenerate colorbar.
    fig = viz.plot_od_demand(braess_scenario().demand)
    assert isinstance(fig, Figure)
    out = tmp_path / "od.png"
    fig.savefig(out, dpi=100)
    plt.close(fig)
    assert out.stat().st_size > 0


def test_plot_flow_scatter_returns_figure():
    n = braess_scenario().network.n_links
    fig = viz.plot_flow_scatter(
        ("bfw", np.array([4.0, 2.0, 2.0, 2.0, 4.0])),
        {"toy": _fake_flows(n, 1.5, 3), "fw": _fake_flows(n, 0.1, 2)},
    )
    assert isinstance(fig, Figure)
    plt.close(fig)


# --- SiouxFalls coordinate parsing (cached data only; gated like conftest) ---
def test_siouxfalls_coordinate_parsing():
    scenario = load_or_skip("siouxfalls")  # skips cleanly when data unavailable
    pos = viz.node_positions(scenario.network)
    assert len(pos) == scenario.network.n_nodes == 24
    # Geographic WGS84 lon/lat from SiouxFalls_node.tntp (node 1 = -96.77, 43.61),
    # NOT the small-integer layered fallback -> confirms the TNTP parse path.
    assert pos[1][0] == pytest.approx(-96.77041974, abs=1e-4)
    assert pos[1][1] == pytest.approx(43.61282792, abs=1e-4)
    assert viz.node_positions(scenario.network) == pos  # deterministic


# --- import guard (matplotlib blocked): mirror the torch/dtalite-blocked test -
def test_import_tabench_silent_when_matplotlib_blocked():
    """`import tabench` must stay green and silent with matplotlib blocked, and
    `tabench.viz` must import (matplotlib-free geometry still works) while its
    plotting calls raise a clear install-hinted error -- run in a subprocess with
    a meta_path blocker."""
    code = (
        "import sys, io, importlib.abc\n"
        "class B(importlib.abc.MetaPathFinder):\n"
        "    def find_spec(self, name, path=None, target=None):\n"
        "        if name == 'matplotlib' or name.startswith('matplotlib.'):\n"
        "            raise ModuleNotFoundError(name, name='matplotlib')\n"
        "        return None\n"
        "sys.meta_path.insert(0, B())\n"
        "for m in [x for x in sys.modules if x == 'matplotlib' or x.startswith('matplotlib.')]:\n"
        "    sys.modules.pop(m, None)\n"
        "buf = io.StringIO(); old = sys.stdout; sys.stdout = buf\n"
        "import tabench\n"
        "from tabench.models import MODEL_REGISTRY\n"
        "import tabench.viz as viz\n"
        "from tabench import braess_scenario\n"
        "sys.stdout = old\n"
        "assert 'aon' in MODEL_REGISTRY, 'numpy core failed to register'\n"
        "assert viz._HAS_MPL is False, 'viz should see matplotlib as absent'\n"
        "pos = viz.node_positions(braess_scenario().network)\n"
        "assert len(pos) == 4, 'node_positions must work without matplotlib'\n"
        "try:\n"
        "    viz.plot_od_demand(braess_scenario().demand)\n"
        "    raise AssertionError('plot_od_demand must raise without matplotlib')\n"
        "except ModuleNotFoundError as e:\n"
        "    assert e.name == 'matplotlib', e.name\n"
        "assert buf.getvalue() == '', 'import tabench polluted stdout: %r' % buf.getvalue()\n"
        "print('GUARD_OK')\n"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert "GUARD_OK" in proc.stdout


# --- demo integration -------------------------------------------------------
def _run_demo(args, cwd):
    return subprocess.run(
        [sys.executable, str(_DEMO), *args],
        capture_output=True,
        text=True,
        cwd=str(cwd),
    )


def test_demo_viz_off_stdout_regression():
    """--viz OFF stdout is unchanged: pin the exact table header + closing
    paragraph, the deterministic first line, the six model rows in order, and the
    censored toy row. (Chosen over full byte-identity because the e-16 gap digits
    are platform/BLAS-dependent -- a full-output compare would be flaky on CI.)"""
    proc = _run_demo([], cwd=_REPO_ROOT)
    assert proc.returncode == 0, proc.stderr
    out = proc.stdout
    assert out.startswith("Scenario: braess (hash cf00f411cdccec88)\n")
    header = f"{'model':<16}{'certified rel. gap':>20}{'feasible':>10}"
    assert header in out
    assert "-" * 46 in out
    # Model rows appear in declaration order.
    order = ["aon", "msa", "fw", "cfw", "bfw", "toy-surrogate"]
    indices = [out.index(name) for name in order]
    assert indices == sorted(indices)
    assert "nan" in out  # the censored toy-surrogate gap
    closing = (
        "\nThe harness certifies every model's gap externally. The black box is"
        "\nscored by the identical certificate as Frank-Wolfe -- and because its"
        "\nflows fail the demand-aware feasibility audit, its gap is censored"
        "\n(nan) rather than scored: garbage can neither crash the experiment"
        "\nnor top the leaderboard."
    )
    assert closing in out
    assert out.rstrip().endswith("nor top the leaderboard.")
    # No viz side effects when the flag is off.
    assert "Saved:" not in out
    assert ".png" not in out


def test_demo_viz_on_writes_three_pngs(tmp_path):
    proc = _run_demo(["--viz", "--viz-out", str(tmp_path)], cwd=_REPO_ROOT)
    assert proc.returncode == 0, proc.stderr
    for name in ("01_od_demand.png", "02_link_flows.png", "03_model_vs_gt.png"):
        png = tmp_path / name
        assert png.exists(), f"missing {name}: {proc.stdout}\n{proc.stderr}"
        assert png.stat().st_size > 0, f"empty {name}"
    assert "P1 story" in proc.stdout
    # M3: aon (certified, feasible=1) is the FARTHEST off-diagonal series, so the P1-story
    # print must name it honestly and NOT claim the certified solvers all cluster.
    assert "aon" in proc.stdout
    assert "certified solvers cluster on it" not in proc.stdout


def test_m10_viz_out_without_viz_warns_but_stdout_unchanged():
    """m10: --viz-out without --viz warns on stderr; stdout stays byte-for-byte the
    OFF-path output (the regression pin above must still hold)."""
    off = _run_demo([], cwd=_REPO_ROOT)
    both = _run_demo(["--viz-out", "/tmp/tabench-nonexistent-viz"], cwd=_REPO_ROOT)
    assert both.returncode == 0, both.stderr
    assert both.stdout == off.stdout  # stdout unchanged
    assert "viz-out is ignored without --viz" in both.stderr


# --- S0a review fix-batch pins ----------------------------------------------
def test_m1_siouxfalls_every_link_renders_nonzero():
    """M1: real WGS84 coordinates must not drop links to over-shrunk arrows — every
    link draws an arrow with nonzero display length (was 6/76 before the fix)."""
    from matplotlib.text import Annotation

    scenario = load_or_skip("siouxfalls")
    net = scenario.network
    fig = viz.plot_network_flows(net, scenario.reference.link_flows)
    fig.canvas.draw()
    ax = fig.axes[0]
    trans = ax.transData
    arrows = [a for a in ax.texts if isinstance(a, Annotation)]
    assert len(arrows) == net.n_links
    for a in arrows:
        p0 = np.asarray(trans.transform(a.xy))
        p1 = np.asarray(trans.transform(a.xyann))
        assert np.hypot(*(p0 - p1)) > 1.0  # > 1 display pixel == visibly drawn
    plt.close(fig)


def _marker_radius_data(ax):
    """Largest node-marker radius in DATA units, from the drawn scatter collections."""
    import math

    inv = ax.transData.inverted()
    dpi = ax.figure.dpi
    rmax = 0.0
    for coll in ax.collections:
        sizes = coll.get_sizes()
        if len(sizes) == 0:
            continue
        r_pt = 0.5 * math.sqrt(float(np.max(sizes)))
        (x0, _), (x1, _) = inv.transform([(0, 0), (r_pt * dpi / 72.0, 0)])
        rmax = max(rmax, abs(x1 - x0))
    return rmax


def _worst_endpoint_gap(net, ax):
    """Max distance (data units) from any arrow endpoint to the node it should touch."""
    from matplotlib.text import Annotation

    pos = viz.node_positions(net)
    arrows = [a for a in ax.texts if isinstance(a, Annotation)]
    worst = 0.0
    for k, a in enumerate(arrows):
        i, j = int(net.init_node[k]), int(net.term_node[k])
        # xy is the arrowhead (toward term j); xyann is the tail (from init i)
        worst = max(
            worst,
            float(np.hypot(*(np.asarray(a.xy) - np.asarray(pos[j])))),
            float(np.hypot(*(np.asarray(a.xyann) - np.asarray(pos[i])))),
        )
    return worst


def test_r1_arrow_endpoints_are_within_marker_scale_of_nodes(monkeypatch):
    """R1: the absolute inset cap pulls long links to their nodes. Every arrow endpoint
    must sit within a small multiple of the node-marker radius of its node (uncapped, the
    outlying Sioux Falls links stop ~7 marker-radii short and read as disconnected)."""
    scenario = load_or_skip("siouxfalls")
    net = scenario.network

    fig = viz.plot_network_flows(net, scenario.reference.link_flows)
    fig.canvas.draw()
    ax = fig.axes[0]
    r = _marker_radius_data(ax)
    capped = _worst_endpoint_gap(net, ax)
    assert capped <= 6.0 * r, f"arrow endpoint {capped / r:.1f} marker-radii from its node"
    plt.close(fig)

    # The cap must be load-bearing: with it effectively disabled the worst gap grows.
    monkeypatch.setattr(viz, "_INSET_CAP_C", 1e9)
    fig2 = viz.plot_network_flows(net, scenario.reference.link_flows)
    fig2.canvas.draw()
    uncapped = _worst_endpoint_gap(net, fig2.axes[0])
    assert uncapped > capped, "the R1 cap did not pull any long link's arrow inward"
    plt.close(fig2)


def test_r1_braess_arrows_are_not_capped():
    """R1: the cap is calibrated to leave the built-ins' proportional trim untouched — on
    Braess every arrow keeps its full 16%-of-link trim (the cap must not shorten it)."""
    net = braess_scenario().network
    fig = viz.plot_network_flows(net, np.array([4.0, 2.0, 2.0, 2.0, 4.0]))
    fig.canvas.draw()
    ax = fig.axes[0]
    capped = _worst_endpoint_gap(net, ax)
    plt.close(fig)
    # The longest Braess link (3->4, length 2) keeps a per-end trim of 0.16*2 = 0.32 when
    # uncapped; the endpoint-to-node distance equals that trim (this link is unpaired, no
    # perpendicular offset), so anything materially below 0.32 means the cap bound.
    assert capped == pytest.approx(0.32, abs=1e-6), (
        f"Braess arrow was capped (endpoint {capped:.4f} from node, expected 0.32)"
    )


def test_r2_figure_matches_data_aspect():
    """R2: with aspect='equal' the figure must follow the DATA aspect so the network fills
    the canvas instead of sitting in a tall/wide empty band."""
    # Unit: _figsize returns dims whose aspect equals the data box aspect.
    w, h = viz._figsize(2.0, 5.0, 24)  # tall data box (h/w = 2.5)
    assert h / w == pytest.approx(2.5, rel=1e-6)
    w2, h2 = viz._figsize(5.0, 2.0, 24)  # wide data box
    assert w2 / h2 == pytest.approx(2.5, rel=1e-6)

    # Rendered Sioux Falls: figure aspect tracks the (taller-than-wide) data aspect.
    scenario = load_or_skip("siouxfalls")
    pos = viz.node_positions(scenario.network)
    x_range, y_range = viz._ranges(pos)
    fig = viz.plot_network_flows(scenario.network, scenario.reference.link_flows)
    fig_w, fig_h = fig.get_size_inches()
    assert fig_h > fig_w  # SiouxFalls data is taller than wide -> a portrait figure
    assert (fig_h / fig_w) == pytest.approx(y_range / x_range, rel=1e-6)
    plt.close(fig)


def test_m2_corrupt_cache_degrades_to_fallback(tmp_path, monkeypatch):
    """M2: a corrupt / NaN-coord / superset cached node file must degrade to the layered
    fallback, never crash or mis-place a network at the wrong coordinates."""
    import dataclasses

    monkeypatch.setenv("TABENCH_CACHE", str(tmp_path))
    node_dir = tmp_path / "siouxfalls"
    node_dir.mkdir(parents=True)
    node_file = node_dir / "SiouxFalls_node.tntp"
    # A 4-node network carrying the registry key name, so _coords_from_cache matches.
    net = dataclasses.replace(braess_scenario().network, name="siouxfalls")
    fallback = viz._layered_positions(net)

    superset = "\n".join(
        ["Node X Y ;"] + [f"{i} {-96.7 - i * 0.01} {43.5 + i * 0.01} ;" for i in range(1, 25)]
    )
    cases = {
        "non-utf8": b"\xff\xfe\x00\x93" * 64,
        "nan-coords": b"1 nan nan ;\n2 nan nan ;\n3 nan nan ;\n4 nan nan ;\n",
        "superset-24-nodes": (superset + "\n").encode(),
    }
    for label, content in cases.items():
        node_file.write_bytes(content)
        assert viz.node_positions(net) == fallback, f"{label} did not fall back"
    # The non-UTF-8 case must be swallowed by _coords_from_cache, not raised.
    node_file.write_bytes(b"\xff\xfe\x00\x93" * 64)
    assert viz._coords_from_cache(net) is None


def test_m4_compare_and_scatter_reject_wrong_length():
    scenario = braess_scenario()
    n = scenario.network.n_links
    with pytest.raises(ValueError, match="shape"):
        viz.compare_models(scenario, {"bad": np.zeros(n + 2)})
    with pytest.raises(ValueError, match="shape"):
        viz.compare_models(scenario, {"ok": np.zeros(n)}, reference=("gt", np.zeros(n - 1)))
    with pytest.raises(ValueError, match="shape"):
        viz.plot_flow_scatter(("gt", np.zeros(n)), {"bad": np.zeros(n + 1)})


def test_m5_no_pyplot_figure_leak():
    """m5: library-style Figures — 25 sequential calls leave the pyplot registry empty."""
    scenario = braess_scenario()
    flows = np.array([4.0, 2.0, 2.0, 2.0, 4.0])
    for _ in range(25):
        viz.plot_network_flows(scenario.network, flows)  # caller "forgets" to close
    assert plt.get_fignums() == []


def test_m6_scatter_includes_negative_reference_min():
    fig = viz.plot_flow_scatter(
        ("gt", np.array([-5.0, 2.0, 3.0, 4.0, 10.0])),
        {"m": np.array([0.0, 2.0, 3.0, 4.0, 10.0])},
    )
    ax = fig.axes[0]
    assert ax.get_xlim()[0] <= -5.0  # negative reference flow is on-canvas, not clipped
    assert ax.get_ylim()[0] <= -5.0
    plt.close(fig)


def test_m7_od_nan_cell_keeps_distinct_finite_colours():
    matrix = np.array([[1.0, np.nan], [2.0, 3.0]])
    fig = viz.plot_od_demand(matrix)
    im = fig.axes[0].images[0]
    lo, hi = im.get_clim()
    assert np.isfinite(lo) and np.isfinite(hi) and hi == 3.0  # finite-masked scale
    # distinct finite demands get distinct colours (not all collapsed dark)
    assert im.norm(1.0) != im.norm(3.0)
    plt.close(fig)


def test_m8_gt_panel_has_visible_background_rectangle():
    from matplotlib.patches import Rectangle

    scenario = braess_scenario()
    n = scenario.network.n_links
    fig = viz.compare_models(
        scenario, {"m": np.zeros(n)}, reference=("gt", np.array([4.0, 2.0, 2.0, 2.0, 4.0]))
    )
    gt_ax = fig.axes[0]  # reference panel is drawn first
    tinted = [
        p for p in gt_ax.patches
        if isinstance(p, Rectangle)
        and matplotlib.colors.to_hex(p.get_facecolor()) == viz._GT_TINT
    ]
    assert tinted, "GT panel must carry a visible tinted background rectangle (not dead code)"
    plt.close(fig)


def test_m11_empty_compare_and_missing_pos_raise():
    scenario = braess_scenario()
    with pytest.raises(ValueError):
        viz.compare_models(scenario, {})
    with pytest.raises(ValueError, match="missing"):
        viz.node_positions(scenario.network, pos={1: (0.0, 0.0)})  # nodes 2,3,4 absent


def test_m11_scatter_series_are_visually_separable():
    """Coincident series (identical flows) must use distinct markers so none is hidden."""
    ref = np.array([4.0, 2.0, 2.0, 2.0, 4.0])
    fig = viz.plot_flow_scatter(
        ("gt", ref), {"fw": ref.copy(), "cfw": ref.copy(), "bfw": ref.copy()}
    )
    ax = fig.axes[0]
    paths = [tuple(c.get_paths()[0].vertices.round(3).ravel()) for c in ax.collections]
    assert len(set(paths)) == len(paths)  # every series has a distinct marker shape
    plt.close(fig)


def test_m13_dense_network_uses_smaller_markers_than_braess():
    """M1/M13: node markers scale down for dense nets so a big fallback stays legible."""
    assert viz._node_scale(4) == 1.0  # Braess unchanged
    assert viz._node_scale(24) < 0.5  # Sioux Falls markers shrink
    assert viz._node_scale(200) == pytest.approx(0.18)  # floored, not vanishing
    assert viz._fig_scale(24) > viz._fig_scale(4)  # bigger canvas for bigger nets
