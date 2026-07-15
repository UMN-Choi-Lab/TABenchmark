"""Adapters wrapping black-box artifacts into the model contract."""

from .callable_adapter import CallableModel

__all__ = ["CallableModel"]

# The SUMO marouter adapter needs the optional ``eclipse-sumo`` wheel
# (``pip install tabench[sumo]``): the numpy/scipy core must import without it.
# Guard the import and swallow ONLY a missing-``sumo`` failure -- any other
# ImportError is a real bug in the module and must propagate. When ``sumo`` is
# absent the model is simply not registered (its @register_model never runs), so
# ``MODEL_REGISTRY``/``tabench list`` lack it and the register-model invariant
# (every registered model is instantiable) is preserved.
try:
    from .sumo_marouter import SumoMarouterModel  # noqa: F401

    _HAS_SUMO = True
except ModuleNotFoundError as exc:  # pragma: no cover - exercised by the sumo-free legs
    if exc.name != "sumo":
        raise
    _HAS_SUMO = False

if _HAS_SUMO:
    __all__.append("SumoMarouterModel")

# The DTALite static-assignment adapter needs the optional ``DTALite`` wheel
# (``pip install tabench[dtalite]``): the numpy/scipy core must import without it.
# Guard the import and swallow ONLY a missing-``DTALite`` failure -- any other
# ImportError is a real bug in the module and must propagate. NOTE the exact case: the
# module is ``DTALite`` (capital D, capital TAL), not ``dtalite``. When absent the model
# is simply not registered (its @register_model never runs), so ``MODEL_REGISTRY`` /
# ``tabench list`` lack it and the register-model invariant holds. The adapter probes
# availability with ``find_spec`` (never ``import DTALite``, which prints a banner and
# ctypes-loads an OpenMP engine into the host), so this import stays stdout-silent.
try:
    from .dtalite_tap import DTALiteTapModel  # noqa: F401

    _HAS_DTALITE = True
except ModuleNotFoundError as exc:  # pragma: no cover - exercised by the dtalite-free legs
    if exc.name != "DTALite":
        raise
    _HAS_DTALITE = False

if _HAS_DTALITE:
    __all__.append("DTALiteTapModel")
