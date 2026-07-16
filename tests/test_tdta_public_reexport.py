"""``tabench.tdta`` public names at the top-level ``tabench`` package (additive,
tutorials batch): every other parallel track (``bottleneck``, ``dta``, ``dnl``,
``newell``, ``transit``, ...) is re-exported from ``tabench/__init__.py`` so a
tutorial can write one `from tabench import (...)` line; ``tdta`` (adr-031) was
the one track left off that list. This locks the re-export in (identity, not a
copy) so a future edit cannot silently drop it again.
"""

import tabench
import tabench.tdta as tdta
from tabench.metrics import TDTAEvaluator


def test_tdta_names_are_re_exported_at_top_level() -> None:
    for name in tdta.__all__:
        assert hasattr(tabench, name), f"tabench.{name} missing (tdta re-export)"
        assert getattr(tabench, name) is getattr(tdta, name)


def test_tdta_evaluator_is_re_exported_at_top_level() -> None:
    """The P1 certifier for the track, mirroring DUEEvaluator/SODTAEvaluator/
    CellSODTAEvaluator already being top-level for the sibling tracks."""
    assert tabench.TDTAEvaluator is TDTAEvaluator
