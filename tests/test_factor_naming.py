"""The T1 inner-loop count factor has ONE canonical name across all native solvers.

B6 unified three historical spellings of the same concept ("iterations of the inner
loop per outer iteration") -- ``inner_rounds`` (algb/oba/tapas), ``inner_sweeps``
(gradient_projection/br_ue/dtd_link), ``inner_iters`` (multiclass/sc_tap/vi_asym) --
onto ``inner_iterations``. This is the standing guard so a fourth variant can never
silently reappear: any registered T1 model that exposes an inner-loop count factor
must name it exactly ``inner_iterations``, with a single explicit allowlist for the
one justified keep (``sumo-marouter`` mirrors the marouter CLI flag
``--max-inner-iterations`` -- engine vernacular, documented "not the repo gap").
"""

from tabench.models.base import MODEL_REGISTRY

CANONICAL = "inner_iterations"

# Justified keep: model name -> the non-canonical inner-loop factor name it may use.
# sumo-marouter forwards the marouter CLI flag --max-inner-iterations verbatim, so its
# factor stays engine-vernacular; this entry is the ONLY sanctioned exception.
_ALLOWLIST = {"sumo-marouter": {"max_inner_iterations"}}

# The nine native T1 solvers whose inner-loop factor B6 renamed to the canonical name.
_RENAMED_NINE = (
    "algb", "oba", "tapas",            # was inner_rounds
    "gp", "br-ue", "dtd-link",         # was inner_sweeps
    "multiclass", "sc-tap", "vi-asym",  # was inner_iters
)


def _is_inner_loop_count(key: str) -> bool:
    """A factor whose name denotes a count of inner-loop iterations/rounds/sweeps."""
    low = key.lower()
    return "inner" in low and any(w in low for w in ("iter", "round", "sweep"))


def test_inner_loop_factor_name_is_canonical():
    """Every registered T1 model that exposes an inner-loop count factor names it
    exactly ``inner_iterations`` (or an allowlisted engine-vernacular keep). Catches a
    reintroduced inner_rounds/inner_sweeps/inner_iters -- or any new fourth spelling."""
    offenders: dict[str, list[str]] = {}
    for name, cls in MODEL_REGISTRY.items():
        allowed = _ALLOWLIST.get(name, set())
        for key in cls.factors:
            if not _is_inner_loop_count(key):
                continue
            if key == CANONICAL or key in allowed:
                continue
            offenders.setdefault(name, []).append(key)
    assert not offenders, (
        f"non-canonical inner-loop factor names (expected {CANONICAL!r} or an "
        f"allowlisted keep {_ALLOWLIST!r}): {offenders}"
    )


def test_the_nine_renamed_natives_expose_canonical_name():
    """Pin the exact nine natives B6 unified by name, so a regression that renamed one
    back to its old spelling is caught explicitly, not only by the generic scan."""
    missing = [n for n in _RENAMED_NINE if CANONICAL not in MODEL_REGISTRY[n].factors]
    assert not missing, f"these natives lost the canonical {CANONICAL!r} factor: {missing}"


def test_allowlist_is_load_bearing_not_vacuous():
    """The allowlist entry must correspond to a real keep: sumo-marouter really does
    expose max_inner_iterations (the --max-inner-iterations mirror) and NOT the
    canonical name, so the exemption cannot silently rot into a blanket pass."""
    for name, allowed in _ALLOWLIST.items():
        factors = MODEL_REGISTRY[name].factors
        for key in allowed:
            assert key in factors, f"stale allowlist: {name} no longer has {key!r}"
        assert CANONICAL not in factors, (
            f"{name} now uses the canonical name; drop its allowlist entry"
        )
