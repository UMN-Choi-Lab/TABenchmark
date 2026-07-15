"""Closed-form tests for SimOpt-style progress curves and solvability profiles.

Every value below is derivable by hand from a tiny staircase or a short list of
solve times; the SimOpt semantics measured against the library source
(docs/design/adr-032) are pinned as edge cases: strict-``<`` crossing at a knot,
censored entries staying in cdf denominators, the flat-zero curve for a
non-finite quantile, union-mesh aggregation, and the byte-deterministic
functional bootstrap. The braess integration regression anchors the α-solve
times {msa: 5, fw: 24, bfw: 4} measured on the certified rows, and the golden
Braess hash is re-asserted (this sprint is additive over the runner).
"""

import json
import math

import numpy as np
import pytest

from tabench import (
    BiconjugateFrankWolfeModel,
    Budget,
    CallableModel,
    FrankWolfeModel,
    MSAModel,
    braess_scenario,
    run_experiment,
)
from tabench.experiments.profiles import (
    Run,
    StepCurve,
    _cdf_from_times,
    _default_metric,
    _quantile_jump,
    bootstrap_progress_band,
    cdf_solvability,
    data_profile,
    diff_profile,
    difference_of_curves,
    load_run,
    mean_of_curves,
    progress_curves,
    quantile_of_curves,
    quantile_solvability,
    read_profiles,
    run_provenance,
    solve_times,
    write_profiles,
)

INF = float("inf")

# The append of the sue_family/profiles machinery must leave every logit
# scenario byte-identical; re-asserted here as the additive-sprint guarantee.
BRAESS_GOLDEN_HASH = "cf00f411cdccec88019979e8cfbf3d8014ba590688b285a1db667315ac96762d"


def _run(rows, budget=None, sue_theta=None, models=None, scenario="toy"):
    """Minimal hand-built :class:`Run` (rows + just enough manifest)."""
    manifest = {
        "scenario": scenario,
        "scenario_hash": "d" * 64,
        "scenario_sue_theta": sue_theta,
        "scenario_sue_family": "probit" if sue_theta is not None else None,
        "budget": budget or {"iterations": None, "sp_calls": None, "wall_seconds": None},
        "seed": 0,
        "macroreps": 1,
        "models": models or {},
        "environment": {"git_commit": "test"},
    }
    return Run(rows, manifest)


def _row(model, sp, rg, feasible=1.0, macrorep=0, iterations=None, **extra):
    it = iterations if iterations is not None else sp
    return {
        "model": model,
        "macrorep": macrorep,
        "iterations": it,
        "sp_calls": sp,
        "wall_ms": 0.0,
        "relative_gap": rg,
        "feasible": feasible,
        **extra,
    }


# ------------------------------------------------------------- StepCurve algebra


def test_lookup_holds_and_censors_before_first_knot():
    c = StepCurve((1.0, 2.0, 3.0), (0.5, 0.2, 0.1))
    assert c.lookup(0.5) == INF  # before the first checkpoint: unsolved (D5)
    assert c.lookup(1.0) == 0.5
    assert c.lookup(1.9) == 0.5
    assert c.lookup(2.0) == 0.2
    assert c.lookup(2.9) == 0.2
    assert c.lookup(3.0) == 0.1
    assert c.lookup(100.0) == 0.1


def test_crossing_time_strict_vs_nonstrict_at_a_knot():
    c = StepCurve((1.0, 2.0, 3.0), (0.5, 0.2, 0.1))
    # threshold exactly at a knot value: strict < excludes it (SimOpt D3),
    # so the first sub-threshold knot is the next one down.
    assert c.crossing_time(0.2, strict=True) == 3.0
    assert c.crossing_time(0.2, strict=False) == 2.0


def test_crossing_time_never_is_inf():
    c = StepCurve((1.0, 2.0, 3.0), (0.5, 0.2, 0.1))
    assert c.crossing_time(0.05) == INF


def test_area_left_endpoint_staircase():
    # 2 over [0,1) + 4 over [1,3); the final y is a right endpoint, not integrated.
    c = StepCurve((0.0, 1.0, 3.0), (2.0, 4.0, 9.0))
    assert c.area() == pytest.approx(2.0 * 1.0 + 4.0 * 2.0)


def test_stepcurve_rejects_bad_mesh():
    with pytest.raises(ValueError):
        StepCurve((1.0, 1.0), (0.0, 0.0))  # not strictly increasing
    with pytest.raises(ValueError):
        StepCurve((1.0, 2.0), (0.0,))  # length mismatch


# ------------------------------------------------------------- solvability cdfs


def test_cdf_terminal_and_censored_denominator():
    # {0.2, 0.5, inf}: two of three solve; inf stays in the denominator (D4).
    c = _cdf_from_times([0.2, 0.5, INF], 1.0)
    assert c.lookup(0.1) == pytest.approx(0.0)
    assert c.lookup(0.2) == pytest.approx(1.0 / 3.0)
    assert c.lookup(0.5) == pytest.approx(2.0 / 3.0)
    assert c.y[-1] == pytest.approx(2.0 / 3.0)  # never reaches 1: the inf is counted


def test_cdf_all_censored_is_flat_zero():
    c = _cdf_from_times([INF, INF], 1.0)
    assert all(v == 0.0 for v in c.y)


# ------------------------------------------------------------- curve aggregation


def test_mean_of_curves_on_union_mesh():
    c1 = StepCurve((0.0, 2.0, 4.0), (1.0, 2.0, 3.0))
    c2 = StepCurve((0.0, 1.0, 4.0), (10.0, 20.0, 30.0))
    m = mean_of_curves([c1, c2])
    assert m.x == (0.0, 1.0, 2.0, 4.0)
    assert m.lookup(0.0) == pytest.approx(5.5)
    assert m.lookup(1.0) == pytest.approx((1.0 + 20.0) / 2.0)
    assert m.lookup(2.0) == pytest.approx((2.0 + 20.0) / 2.0)
    assert m.lookup(4.0) == pytest.approx((3.0 + 30.0) / 2.0)


def test_quantile_of_curves_matches_simopt_exclusive():
    # SimOpt's estimator is statistics.quantiles(n=100)[int(beta*99)] (exclusive),
    # not np.quantile linear; cross-checked here against the stdlib it ports.
    from statistics import quantiles

    c1 = StepCurve((0.0, 1.0), (1.0, 1.0))
    c2 = StepCurve((0.0, 1.0), (3.0, 3.0))
    for beta in (0.25, 0.5, 0.75):
        expected = quantiles([1.0, 3.0], n=100)[int(beta * 99)]
        assert quantile_of_curves([c1, c2], beta).lookup(0.0) == pytest.approx(expected)
    # the censoring-robust opt-in is the type-1 lower inverted-cdf value
    got = quantile_of_curves([c1, c2], 0.25, method="censoring-robust").lookup(0.0)
    assert got == pytest.approx(1.0)


def test_quantile_of_curves_inf_is_never_nan():
    # censored (+inf) pointwise values must yield +inf, never a np.quantile NaN (M3).
    c1 = StepCurve((0.0, 1.0), (1.0, 1.0))
    cens = StepCurve((0.0, 1.0), (INF, INF))
    q = quantile_of_curves([c1, cens, cens], 0.75)
    assert all(v == INF for v in q.y)


# ------------------------------------------------------- quantile solvability jump


def test_quantile_jump_simopt_finite():
    # SimOpt exclusive estimator: {0.2,0.5,0.9,inf} at beta=0.5 -> 0.7, a finite jump.
    from statistics import quantiles

    assert quantiles([0.2, 0.5, 0.9, INF], n=100)[int(0.5 * 99)] == pytest.approx(0.7)
    c = _quantile_jump([0.2, 0.5, 0.9, INF], 0.5, 1.0)
    assert c.lookup(0.69) == 0.0
    assert c.lookup(0.7) == 1.0


def test_quantile_jump_simopt_flat_zero_reaching_censored_tail():
    # {0.2,0.5,inf} beta=0.5: statistics.quantiles interpolates INTO the censored
    # tail -> NaN -> flat-zero. This is exactly the case the old type-1 estimator
    # flattered (it jumped at 0.5); default is now SimOpt-exact (adr-032 D9).
    from statistics import quantiles

    assert math.isnan(quantiles([0.2, 0.5, INF], n=100)[int(0.5 * 99)])
    assert _quantile_jump([0.2, 0.5, INF], 0.5, 1.0).y == (0.0, 0.0)
    # beta=0.9 -> the quantile is +inf -> also flat-zero.
    assert _quantile_jump([0.2, 0.5, INF], 0.9, 1.0).y == (0.0, 0.0)


def test_quantile_jump_censoring_robust_opt_in_diverges():
    from statistics import quantiles

    # {0.2,0.5,inf} beta=0.5: SimOpt flat-zero (above) vs censoring-robust jump at 0.5.
    rob = _quantile_jump([0.2, 0.5, INF], 0.5, 1.0, method="censoring-robust")
    assert rob.lookup(0.49) == 0.0 and rob.lookup(0.5) == 1.0
    # {0.1,0.2,0.3,inf} beta=0.5: SimOpt jumps at 0.25, censoring-robust at 0.2.
    assert quantiles([0.1, 0.2, 0.3, INF], n=100)[int(0.5 * 99)] == pytest.approx(0.25)
    sim = _quantile_jump([0.1, 0.2, 0.3, INF], 0.5, 1.0)
    rob2 = _quantile_jump([0.1, 0.2, 0.3, INF], 0.5, 1.0, method="censoring-robust")
    assert next(x for x, y in zip(sim.x, sim.y, strict=True) if y == 1.0) == pytest.approx(0.25)
    assert next(x for x, y in zip(rob2.x, rob2.y, strict=True) if y == 1.0) == pytest.approx(0.2)


def test_quantile_solvability_public_jump():
    # one macrorep solving at sp=2 with a realized budget of 4 -> normalized 0.5;
    # the beta-quantile of a single solve time is that time -> a 0->1 jump at 0.5.
    run = _run([_row("A", 1, 0.5), _row("A", 2, 0.0), _row("A", 4, 0.0)], scenario="s1")
    q = quantile_solvability(run, alpha=1e-4, beta=0.5, axis="sp_calls")
    assert q["A"].lookup(0.4) == pytest.approx(0.0)
    assert q["A"].lookup(0.5) == pytest.approx(1.0)


# ------------------------------------------------------------- Moré-Wild profiles


def test_data_profile_staircase_across_two_scenarios():
    tau = 0.1
    s1 = _run(
        [
            _row("A", 1, 0.5), _row("A", 2, 0.05),
            _row("B", 1, 0.5), _row("B", 2, 0.3), _row("B", 4, 0.05),
        ],
        scenario="s1",
    )
    s2 = _run(
        [
            _row("A", 3, 0.5), _row("A", 6, 0.05),
            _row("B", 3, 0.5), _row("B", 6, 0.3),  # B never reaches tau in s2
        ],
        scenario="s2",
    )
    dp = data_profile([s1, s2], tau=tau, axis="sp_calls", work_unit=1.0)
    # A solves both problems (work 2, 6); B solves one of two (work 4, inf).
    assert dp["A"].lookup(2.0) == pytest.approx(0.5)
    assert dp["A"].lookup(6.0) == pytest.approx(1.0)
    assert dp["B"].lookup(4.0) == pytest.approx(0.5)
    assert dp["B"].y[-1] == pytest.approx(0.5)  # the never-solved problem stays in denom


def test_data_profile_work_unit_scales_kappa():
    s1 = _run([_row("A", 1, 0.5), _row("A", 4, 0.05)], scenario="s1")
    unit1 = data_profile([s1], tau=0.1, axis="sp_calls", work_unit=1.0)
    unit2 = data_profile([s1], tau=0.1, axis="sp_calls", work_unit=2.0)
    # work 4 -> kappa 4 (unit 1) vs kappa 2 (unit 2).
    assert unit1["A"].lookup(3.9) == pytest.approx(0.0)
    assert unit1["A"].lookup(4.0) == pytest.approx(1.0)
    assert unit2["A"].lookup(1.9) == pytest.approx(0.0)
    assert unit2["A"].lookup(2.0) == pytest.approx(1.0)


# ------------------------------------------------------------- difference profiles


def test_diff_profile_against_self_is_zero():
    profile = {"A": StepCurve((0.0, 1.0), (0.2, 0.6)), "B": StepCurve((0.0, 1.0), (0.5, 0.9))}
    diff = diff_profile(profile, "A")
    assert all(v == 0.0 for v in diff["A"].y)


def test_diff_profile_values_on_union_mesh():
    profile = {"A": StepCurve((0.0, 1.0), (0.2, 0.6)), "B": StepCurve((0.0, 1.0), (0.5, 0.9))}
    diff = diff_profile(profile, "A")
    assert diff["B"].lookup(0.0) == pytest.approx(0.3)
    assert diff["B"].lookup(1.0) == pytest.approx(0.3)


# --------------------------------------------------------- functional bootstrap


def test_bootstrap_band_is_byte_deterministic():
    curves = [StepCurve((0.0, 1.0), (0.1 * k, 0.2 * k)) for k in (1, 2, 3, 4)]
    lo1, hi1 = bootstrap_progress_band(curves, root_seed=7, b=2000)
    lo2, hi2 = bootstrap_progress_band(curves, root_seed=7, b=2000)
    assert lo1.y == lo2.y and hi1.y == hi2.y


def test_bootstrap_band_zero_width_when_macroreps_identical():
    c = StepCurve((0.0, 1.0, 2.0), (0.9, 0.4, 0.1))
    lo, hi = bootstrap_progress_band([c, c, c], root_seed=3, b=500)
    assert lo.y == hi.y  # zero width: identical macroreps -> no spread
    assert lo.y == pytest.approx(c.y)  # centered on the common curve (mean fp only)


def test_bootstrap_band_brackets_the_mean_curve():
    curves = [StepCurve((0.0, 1.0), (0.1 * k, 0.2 * k)) for k in (1, 2, 3, 4)]
    mean = mean_of_curves(curves)
    lo, hi = bootstrap_progress_band(curves, root_seed=1, b=4000)
    for t in mean.x:
        assert lo.lookup(t) <= mean.lookup(t) <= hi.lookup(t)


# ------------------------------------------------------------- censoring / axes


def test_infeasible_and_nan_metric_are_censored_but_stay_in_denominator():
    # A: one macrorep solves at sp=2, the other is infeasible (feasible=0);
    # a third checkpoint carries a finite-feasible but NaN metric -> never solved.
    run = _run(
        [
            _row("A", 1, 0.5, macrorep=0), _row("A", 2, 0.0, macrorep=0),
            _row("A", 1, 0.5, feasible=0.0, macrorep=1),
            _row("A", 2, 0.5, feasible=0.0, macrorep=1),
        ],
        budget={"iterations": None, "sp_calls": None, "wall_seconds": None},
        scenario="s1",
    )
    cdf = cdf_solvability(run, alpha=1e-4, axis="sp_calls")
    # 1 of 2 macroreps solved; the censored one stays in the denominator (D4).
    assert cdf["A"].y[-1] == pytest.approx(0.5)


def test_metric_nan_row_never_solves():
    run = _run([_row("A", 1, 0.5), _row("A", 2, float("nan"))], scenario="s1")
    curves = progress_curves(run, axis="sp_calls")
    assert curves[("A", 0)].crossing_time(1e-4) == INF


def test_cross_scenario_mean_of_cdf_curves():
    s1 = _run([_row("A", 1, 0.5), _row("A", 2, 0.0)], scenario="s1")  # solves
    s2 = _run([_row("A", 1, 0.5), _row("A", 2, 0.5)], scenario="s2")  # never solves
    cdf = cdf_solvability([s1, s2], alpha=1e-4, axis="sp_calls")
    # mean of a terminal-1 cdf and a terminal-0 cdf.
    assert cdf["A"].y[-1] == pytest.approx(0.5)


def test_zero_axis_refused_for_finite_model_but_not_for_censored_one():
    # A does real work yet reports sp_calls=0 everywhere: the axis is degenerate,
    # refuse (D2). A censored-only model at sp=0 is exempt (it is +inf regardless).
    finite = _run([_row("A", 0, 0.5), _row("A", 0, 0.0)], scenario="s1")
    with pytest.raises(ValueError):
        progress_curves(finite, axis="sp_calls")
    censored = _run(
        [_row("A", 2, 0.0), _row("Z", 0, float("nan"), feasible=0.0)],
        scenario="s1",
    )
    curves = progress_curves(censored, axis="sp_calls")  # no raise
    assert curves[("Z", 0)].crossing_time(1e-4) == INF


def test_wall_ms_refused_by_ranked_profile_but_allowed_descriptively():
    run = _run([_row("A", 1, 0.5, wall_ms=1.0), _row("A", 2, 0.0, wall_ms=2.0)], scenario="s1")
    progress_curves(run, axis="wall_ms")  # descriptive: allowed
    with pytest.raises(ValueError):
        cdf_solvability(run, alpha=1e-4, axis="wall_ms")  # ranked: refused (P6)


def test_default_metric_read_from_manifest():
    sue = _run([_row("A", 1, 0.5)], sue_theta=0.5, scenario="s1")
    so_models = {"so-bfw": {"capabilities": {"paradigm": "static_so"}}}
    so = _run(
        [{"model": "so-bfw", "macrorep": 0, "iterations": 1, "sp_calls": 1, "wall_ms": 0.0,
          "so_relative_gap": 0.0, "feasible": 1.0}],
        models=so_models,
        scenario="s1",
    )
    assert _default_metric(sue.manifest) == "sue_fixed_point_residual"
    assert _default_metric(so.manifest) == "so_relative_gap"
    assert _default_metric(_run([], scenario="s1").manifest) == "relative_gap"


# --------------------------------------------------- braess integration regression


@pytest.fixture(scope="module")
def braess_run():
    scenario = braess_scenario()
    result = run_experiment(
        scenario,
        [MSAModel(), FrankWolfeModel(), BiconjugateFrankWolfeModel()],
        Budget(iterations=50),
        seed=0,
    )
    return Run.from_result(result)


def test_braess_alpha_solve_times_regression(braess_run):
    curves = progress_curves(braess_run, axis="sp_calls")
    by_model = {model: t for (model, _mr), t in solve_times(curves, 1e-4).items()}
    assert by_model == {"msa": 5.0, "fw": 24.0, "bfw": 4.0}


def test_braess_early_stopped_trace_carries_to_budget_end(braess_run):
    curves = progress_curves(braess_run, axis="sp_calls")
    bfw = curves[("bfw", 0)]
    # bfw converges in 3 checkpoints (sp 2,3,4); the final value carries to the
    # realized budget end (max sp across the grid = 51), D5.
    assert bfw.x[:3] == (2.0, 3.0, 4.0)
    assert bfw.x[-1] == 51.0
    assert bfw.lookup(51.0) == pytest.approx(bfw.lookup(4.0))


def test_braess_golden_hash_preserved():
    assert braess_scenario().content_hash() == BRAESS_GOLDEN_HASH


# ------------------------------------------------------------- artifact round-trip


def test_profiles_json_roundtrip_and_schema(braess_run, tmp_path):
    cdf = cdf_solvability(braess_run, 1e-4, axis="sp_calls")
    protocol = {"metric": "relative_gap", "axis": "sp_calls", "alpha": 1e-4, "crossing": "strict-<"}
    out = tmp_path / "profiles.json"
    write_profiles(out, {"cdf_solvability": cdf}, protocol, run_provenance(braess_run))
    doc, back = read_profiles(out)
    # curves survive byte-identically through the artifact.
    for model, curve in cdf.items():
        assert back["cdf_solvability"][model].x == curve.x
        assert back["cdf_solvability"][model].y == curve.y
    # the artifact carries the protocol constants and the scenario hash provenance.
    assert doc["schema"] == "tabench-profiles-v1"
    assert doc["protocol"]["crossing"] == "strict-<"
    assert doc["provenance"][0]["scenario_hash"] == BRAESS_GOLDEN_HASH


def test_in_memory_and_on_disk_runs_give_identical_profiles(tmp_path):
    scenario = braess_scenario()
    result = run_experiment(
        scenario,
        [MSAModel(), FrankWolfeModel()],
        Budget(iterations=20),
        seed=0,
        out_dir=tmp_path,
    )
    in_memory = cdf_solvability(Run.from_result(result), 1e-4, axis="sp_calls")
    (csv_path,) = tmp_path.glob("*.csv")
    on_disk = cdf_solvability(load_run(csv_path), 1e-4, axis="sp_calls")
    for model in in_memory:
        assert on_disk[model].x == in_memory[model].x
        assert on_disk[model].y == in_memory[model].y


def test_censored_black_box_included_as_inf(tmp_path):
    scenario = braess_scenario()

    def naive_surrogate(s, rng):
        base = s.demand.total / 2.0
        return np.abs(base + rng.normal(0.0, 0.5, s.network.n_links))

    result = run_experiment(
        scenario,
        [FrankWolfeModel(), CallableModel(fn=naive_surrogate, name="toy", seedable=True)],
        Budget(iterations=50),
        seed=0,
    )
    # the surrogate discloses sp_calls=0 AND is infeasible: censored, never an
    # axis-abuse refusal; it enters the cdf denominator as never-solved.
    cdf = cdf_solvability(Run.from_result(result), 1e-4, axis="sp_calls")
    assert cdf["toy"].y[-1] == pytest.approx(0.0)
    assert cdf["fw"].y[-1] == pytest.approx(1.0)


# ----------------------------------------------- adversarial-review regressions


def test_incongruent_model_sets_refused(tmp_path):
    # M1/D10: B is run only on s1; without the full-cross-design guard B (tested on
    # one easy scenario) would top the honest A (tested on both).
    s1 = _run([_row("A", 2, 0.0), _row("B", 2, 0.0)], scenario="s1")
    s2 = _run([_row("A", 2, 0.0)], scenario="s2")
    with pytest.raises(ValueError, match="full cross design"):
        cdf_solvability([s1, s2], 1e-4, axis="sp_calls")
    with pytest.raises(ValueError, match="full cross design"):
        data_profile([s1, s2], tau=0.1, axis="sp_calls", work_unit=1.0)


def test_quantile_solvability_flat_zero_matches_simopt_not_flattered():
    # M2: two-macrorep median where only 2/4 solve -> SimOpt flat-zero; the shipped
    # type-1 default used to flatter this to a jump at 0.5.
    run = _run(
        [
            _row("A", 2, 0.0, macrorep=0), _row("A", 5, 0.0, macrorep=1),
            _row("A", 2, 0.5, feasible=0.0, macrorep=2),
            _row("A", 2, 0.5, feasible=0.0, macrorep=3),
        ],
        scenario="s1",
    )
    q = quantile_solvability(run, alpha=1e-4, beta=0.5, axis="sp_calls")
    assert all(v == 0.0 for v in q["A"].y)  # flat-zero (SimOpt), not a 0->1 jump
    q_robust = quantile_solvability(
        run, alpha=1e-4, beta=0.5, axis="sp_calls", quantile_method="censoring-robust"
    )
    assert any(v == 1.0 for v in q_robust["A"].y)  # the opt-in still jumps


def test_difference_of_curves_infinities():
    # M3: both-censored -> zero; one-censored -> +/-inf; never NaN.
    fin = StepCurve((0.0, 1.0), (1.0, 1.0))
    cens = StepCurve((0.0, 1.0), (INF, INF))
    assert all(v == 0.0 for v in difference_of_curves(cens, cens).y)
    assert all(v == INF for v in difference_of_curves(cens, fin).y)
    assert all(v == -INF for v in difference_of_curves(fin, cens).y)


def test_bootstrap_band_inf_honest_not_nan():
    # M3: a resample mean touching a censored +inf stays +inf, never a np.percentile NaN.
    fin = StepCurve((0.0, 1.0), (0.5, 0.5))
    cens = StepCurve((0.0, 1.0), (INF, INF))
    lo, hi = bootstrap_progress_band([fin, cens, cens], root_seed=0, b=500)
    assert not any(math.isnan(v) for v in lo.y + hi.y)
    assert hi.y[-1] == INF  # the upper band is honestly censored


def test_write_profiles_refuses_nan(tmp_path):
    # M3: a NaN never reaches the artifact — the curve encoder refuses it, and the
    # allow_nan=False dump refuses one that sneaks into the protocol/provenance.
    from tabench.experiments.profiles import _encode_y

    with pytest.raises(ValueError, match="NaN"):
        _encode_y(float("nan"))
    good = {"cdf_solvability": {"A": StepCurve((0.0, 1.0), (0.5, 0.5))}}
    with pytest.raises(ValueError):
        write_profiles(tmp_path / "p.json", good, {"bad": float("nan")}, [])


def test_axis_guard_uses_requested_metric_both_directions():
    # M4: a learned model does real work but discloses sp_calls=0; on an alternate
    # metric where it is finite the axis is refused; where it is censored, it is not.
    finite_on_alt = _run(
        [
            _row("fw", 2, 0.5, average_excess_cost=0.5),
            _row("fw", 24, 0.0, average_excess_cost=0.0),
            _row("learned", 0, float("nan"), average_excess_cost=0.0),
        ],
        scenario="s1",
    )
    with pytest.raises(ValueError, match="0 at every checkpoint"):
        cdf_solvability(finite_on_alt, 1e-4, axis="sp_calls", metric="average_excess_cost")
    # censored under the requested metric -> exempt, no spurious refusal
    cdf = cdf_solvability(finite_on_alt, 1e-4, axis="sp_calls", metric="relative_gap")
    assert cdf["learned"].y[-1] == pytest.approx(0.0)


def test_mixed_so_grid_keeps_relative_gap_default():
    # M5: one static_so model must not flip the default metric for the UE solvers.
    models = {
        "msa": {"capabilities": {"paradigm": "static_ue"}},
        "so-bfw": {"capabilities": {"paradigm": "static_so"}},
    }
    mixed = _run([_row("msa", 2, 0.5)], models=models, scenario="s1")
    all_so = _run(
        [_row("so-bfw", 2, 0.5)],
        models={"so-bfw": {"capabilities": {"paradigm": "static_so"}}},
        scenario="s1",
    )
    assert _default_metric(mixed.manifest) == "relative_gap"
    assert _default_metric(all_so.manifest) == "so_relative_gap"


def test_t2_estimation_run_raises_clear_error(tmp_path):
    # M6: profiles are the assignment track; a T2 CSV raises a named limitation, not
    # a bare KeyError('model').
    csv_path = tmp_path / "t2-run.csv"
    csv_path.write_text(
        "scenario,estimator,macrorep,iterations,sp_calls,wall_ms,od_feasible,heldout_count_rmse\n"
        "braess,gls,0,1,2,0.0,1,3.5\n"
    )
    with pytest.raises(ValueError, match="T2 estimation"):
        load_run(csv_path)
    # the in-memory path guards too (attack1's KeyError('model') is gone)
    t2 = Run(
        [{"estimator": "gls", "macrorep": 0, "sp_calls": 2, "od_feasible": 1,
          "heldout_count_rmse": 3.5}],
        {"scenario": "braess", "budget": {"sp_calls": None}, "models": {}},
    )
    with pytest.raises(ValueError, match="T2 estimation"):
        cdf_solvability(t2, 1e-4, axis="sp_calls")


def test_overshoot_crossing_censored_consistently():
    # M7: a crossing beyond the manifest budget envelope is censored (+inf) for BOTH
    # the cdf and the quantile jump -- never counted by one and dropped by the other.
    run = _run(
        [_row("A", 5, 0.5), _row("A", 11, 0.0)],
        budget={"iterations": None, "sp_calls": 10},
        scenario="s1",
    )
    cdf = cdf_solvability(run, 1e-4, axis="sp_calls")
    q = quantile_solvability(run, 1e-4, 0.5, axis="sp_calls")
    assert cdf["A"].y[-1] == pytest.approx(0.0)
    assert all(v == 0.0 for v in q["A"].y)


def test_work_unit_dict_missing_scenario_key_refused():
    # M8c: a missing scenario key must not silently fall back to 1 and mix units.
    run = _run([_row("A", 2, 0.0)], scenario="braess")
    with pytest.raises(ValueError, match="no entry for scenario"):
        data_profile([run], tau=0.1, axis="sp_calls", work_unit={"sioux-falls": 24.0})


def test_bootstrap_band_refuses_single_macrorep():
    # M8a: a single curve has no sampling spread; mirror bootstrap_ci's size>1 gate.
    c = StepCurve((0.0, 1.0), (0.5, 0.5))
    with pytest.raises(ValueError, match=">= 2"):
        bootstrap_progress_band([c], root_seed=0)


def test_deterministic_iterations_tiebreak_for_duplicate_axis():
    # M8d: two checkpoints share sp_calls; the higher-iterations row (newer
    # recommendation) wins regardless of CSV row order.
    forward = _run(
        [_row("A", 2, 0.9, iterations=1), _row("A", 2, 0.1, iterations=2)], scenario="s1"
    )
    reverse = _run(
        [_row("A", 2, 0.1, iterations=2), _row("A", 2, 0.9, iterations=1)], scenario="s1"
    )
    cf = progress_curves(forward, axis="sp_calls")[("A", 0)]
    cr = progress_curves(reverse, axis="sp_calls")[("A", 0)]
    assert cf.lookup(2.0) == pytest.approx(0.1)  # iterations=2 wins in both orders
    assert cr.lookup(2.0) == pytest.approx(0.1)


def test_blank_axis_cell_and_nan_mesh_rejected():
    # M8e: a blank/non-finite work coordinate on a checkpoint raises; StepCurve
    # refuses a NaN knot outright.
    blank = _run([_row("A", "", 0.0)], scenario="s1")
    with pytest.raises(ValueError, match="blank/non-finite"):
        progress_curves(blank, axis="sp_calls")
    with pytest.raises(ValueError, match="finite"):
        StepCurve((1.0, float("nan")), (0.0, 0.0))


def test_unknown_metric_name_rejected():
    # M8f: a metric that is not a column raises rather than a silent flat-zero profile.
    run = _run([_row("A", 2, 0.0)], scenario="s1")
    with pytest.raises(ValueError, match="unknown metric"):
        progress_curves(run, metric="not_a_column", axis="sp_calls")


def test_artifact_is_strict_rfc8259_with_infinity_string(tmp_path):
    # M8b: a censored +inf is the JSON string "Infinity", so a strict parser (one
    # that rejects bare Infinity/NaN tokens) can read the artifact.
    profile = {"cdf_solvability": {"A": StepCurve((0.0, 1.0), (0.5, INF))}}
    out = tmp_path / "profiles.json"
    write_profiles(out, profile, {"metric": "relative_gap"}, [])
    text = out.read_text()
    assert "Infinity" in text and '"Infinity"' in text  # the quoted string form

    def _reject_constants(value):
        raise AssertionError(f"non-RFC-8259 constant {value!r} in artifact")

    strict = json.loads(text, parse_constant=_reject_constants)  # would raise on bare tokens
    assert strict["profiles"]["cdf_solvability"]["A"]["y"][1] == "Infinity"
    _, back = read_profiles(out)
    assert back["cdf_solvability"]["A"].y == (0.5, INF)  # round-trips to +inf
