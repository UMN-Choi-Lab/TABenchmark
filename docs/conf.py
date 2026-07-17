"""Sphinx configuration for the TABenchmark documentation site.

The Sphinx source root is a repo-root-shaped MIRROR (assembled by
``docs/build_site.py`` into ``docs/_build/site/``), NOT ``docs/`` itself. The
tutorial notebooks and the README link to sibling repository files with
repo-root-relative paths -- ``../../docs/MODELS.md`` (89 notebooks),
``../../README.md`` (53), ``../../CONTRIBUTING.md`` (27), ``../../docs/design/
adr-*.md``, ... -- which resolve ONLY when the source root mirrors the
repository layout. Building from ``docs/`` directly shifts every notebook one
directory deeper and breaks all of them (measured: ~130 unresolved-link
warnings). So the build is two steps (see ``.readthedocs.yaml`` and the ci.yml
``docs`` job)::

    python docs/build_site.py
    python -m sphinx -T -W --keep-going -c docs -b html docs/_build/site <out>

``-c docs`` points Sphinx at THIS conf.py while the positional source dir is the
mirror. Everything renders with ZERO edits to the real files; the only in-repo
change the site required is the handful of notebook links that pointed into
NON-documentation source (``tests/test_braess.py``, ``demos/demo_profiles.py``),
rewritten to GitHub blob URLs because a relative link into source can never
resolve on a docs site.

Locale note: the notebook execution kernel needs a UTF-8 locale -- a C locale
makes the build die on the first non-ASCII cell. RTD/CI set ``LC_ALL=C.UTF-8``.
"""

from __future__ import annotations

import json
from importlib.util import find_spec
from pathlib import Path

_CONF = Path(__file__).resolve().parent  # <repo>/docs
_REPO = _CONF.parent  # <repo>

# -- Project ----------------------------------------------------------------
project = "TABenchmark"
author = "UMN Choi Lab"
copyright = "2026, UMN Choi Lab"  # noqa: A001 (Sphinx's documented config name)

# -- Extensions -------------------------------------------------------------
extensions = [
    "myst_nb",  # Markdown pages (via myst-parser) + notebook execution at build
    "sphinx.ext.autodoc",  # API reference from docstrings
    "sphinx.ext.napoleon",  # NumPy/Google docstring style
    "sphinx.ext.viewcode",  # [source] links next to documented objects
    "sphinx_copybutton",  # copy button on code blocks
    "sphinxcontrib.mermaid",  # the model-evolution diagram fence in docs/MODELS.md
]

html_theme = "furo"
html_title = "TABenchmark"

source_suffix = {".md": "myst-nb", ".ipynb": "myst-nb"}
root_doc = "index"
exclude_patterns = ["_build", "**/.ipynb_checkpoints"]

# -- MyST -------------------------------------------------------------------
myst_enable_extensions = ["colon_fence", "dollarmath", "attrs_inline"]
# Render the repo's raw ```mermaid fence (docs/MODELS.md) as a diagram, unedited.
myst_fence_as_directive = ["mermaid"]
myst_heading_anchors = 3

# -- Notebook execution (myst-nb) -------------------------------------------
# The site is the named rendering venue for the tutorials (adr-035): notebooks
# are committed stripped and execute at build so their certified prints and
# figures are real. ``cache`` reuses unchanged results on incremental local
# rebuilds; a fresh RTD/CI build executes every non-excluded notebook.
nb_execution_mode = "cache"
nb_execution_timeout = 600  # some solver notebooks exceed the 30 s default
nb_execution_raise_on_error = True  # a broken tutorial must fail the build
# Merge consecutive stdout/stderr into one block per cell: kernels chunk a stream
# nondeterministically across runs, which would otherwise make a few executed pages
# differ byte-for-byte between two cold builds.
nb_merge_streams = True

# Engine-gated tutorials (metadata.tabench.requires_extra in {torch, sumo,
# dtalite, matsim}) cannot execute in the docs environment, which installs only
# ``.[docs,viz]`` -- so exclude them from EXECUTION. They still render as
# readable, un-executed pages. The list is DERIVED from the notebooks' own
# metadata, so it tracks the tree automatically for every KNOWN extra; a NEW
# extra must be added to ``_EXTRA_MODULE`` below, which the loop enforces with
# a loud, self-describing error rather than a bare KeyError. An extra that IS
# available is NOT excluded -- installing sumo/dtalite (or addressing a matsim
# toolchain) un-excludes their notebooks by config alone (only torch's multi-GB
# CUDA wheel stays stuck out). torch/sumo/dtalite probe by module name via
# find_spec (never imported); 'matsim' is a Java-only engine with no python
# module (adr-039), so its probe is the adapter's side-effect-free availability
# CALLABLE (tabench is pip-installed in the docs env, and the adapter module
# imports without any optional extra).
from tabench.models.adapters._matsim_io import matsim_available as _matsim_available

_EXTRA_MODULE = {
    "torch": "torch",
    "sumo": "sumo",
    "dtalite": "DTALite",
    "matsim": _matsim_available,
}


def _extra_available(extra: str) -> bool:
    probe = _EXTRA_MODULE[extra]
    if callable(probe):
        return bool(probe())
    return find_spec(probe) is not None


nb_execution_excludepatterns: list[str] = []
for _nb in sorted((_REPO / "tutorials").rglob("*.ipynb")):
    if ".ipynb_checkpoints" in _nb.parts:
        continue
    _extra = (
        json.loads(_nb.read_text())
        .get("metadata", {})
        .get("tabench", {})
        .get("requires_extra")
    )
    if not _extra:
        continue
    if _extra not in _EXTRA_MODULE:
        raise KeyError(
            f"{_nb.relative_to(_REPO)}: metadata.tabench.requires_extra={_extra!r} has no "
            f"import-probe mapping. Add {_extra!r} to _EXTRA_MODULE in {Path(__file__).name} "
            f"(known: {sorted(_EXTRA_MODULE)})."
        )
    if not _extra_available(_extra):
        nb_execution_excludepatterns.append(f"**/{_nb.parent.name}/{_nb.name}")

# -- autodoc ----------------------------------------------------------------
# tabench is pip-installed in the docs env, so the public packages import
# without a sys.path hack; every one is import-safe without the optional extras
# (the core CI matrix proves `import tabench` needs no torch/sumo/dtalite).
autodoc_default_options = {"members": True, "show-inheritance": True}
autodoc_typehints = "description"
autodoc_member_order = "bysource"
