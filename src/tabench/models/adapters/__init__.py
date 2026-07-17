"""Adapters wrapping black-box artifacts into the model contract."""

from .callable_adapter import CallableModel

# The DTALite SIMULATION adapter (the EDOC row, adr-040) re-exports
# UNCONDITIONALLY — a named deviation from dtalite_tap's module-scope guard:
# the module NEVER imports DTALite in-host (the engine runs only in throwaway
# subprocesses; availability is a find_spec probe / the runtime G0 version
# read), so it imports stdlib+numpy everywhere and its engine-free test half
# runs on the core matrix legs. Not in MODEL_REGISTRY (EDOC producer).
from .dtalite_simulation import DTALiteSimulationAdapter

# The MATSim EDOC adapter needs NO optional python package (the engine is
# Java-only, addressed at runtime via TABENCH_MATSIM_HOME — adr-039), so unlike
# sumo/dtalite the re-export is UNCONDITIONAL: the module imports stdlib+numpy
# everywhere and engine absence surfaces as the matsim_available() probe / a G0
# RAISE, never an ImportError. Not in MODEL_REGISTRY (EDOC producers are not
# the static gap-certified track — the adr-037 precedent).
from .matsim_edoc import MatsimAdapter

__all__ = ["CallableModel", "DTALiteSimulationAdapter", "MatsimAdapter"]

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
