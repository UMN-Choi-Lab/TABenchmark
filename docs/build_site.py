"""Assemble the Sphinx source tree for the TABenchmark documentation site.

The tutorial notebooks and the README link to sibling repository files with
repo-root-relative paths (``../../docs/MODELS.md``, ``../../README.md``, ...),
so the Sphinx source root must mirror the repository layout for those links to
resolve. This script copies the rendered subset of the repo into
``docs/_build/site/`` (git-ignored) with the mapping

    <repo>/README.md          -> docs/_build/site/README.md
    <repo>/CONTRIBUTING.md     -> docs/_build/site/CONTRIBUTING.md
    <repo>/CITATION.cff        -> docs/_build/site/CITATION.cff   (README link target)
    <repo>/LICENSE             -> docs/_build/site/LICENSE        (README link target)
    <repo>/docs/<content>      -> docs/_build/site/docs/<content>
    <repo>/tutorials/          -> docs/_build/site/tutorials/
    <repo>/docs/index.md       -> docs/_build/site/index.md       (site landing page)
    <repo>/docs/api/           -> docs/_build/site/api/           (autodoc pages)

so ``docs/`` and ``tutorials/`` stay the single source of truth for
``git clone`` users. Idempotent: the mirror is wiped and rebuilt each run. See
``docs/conf.py`` for why the build is two steps.
"""

from __future__ import annotations

import shutil
from pathlib import Path

_DOCS = Path(__file__).resolve().parent  # <repo>/docs
_REPO = _DOCS.parent  # <repo>
_SITE = _DOCS / "_build" / "site"

# docs/ entries that are Sphinx MACHINERY, not repo documentation content: they
# must not be copied under the mirror's docs/ (index.md and api/ go to the mirror
# ROOT instead; conf.py/build_site.py/_build/_static are build-time only).
_DOCS_MACHINERY = {
    "conf.py",
    "build_site.py",
    "index.md",
    "api",
    "_build",
    "_static",
    "_templates",
    "__pycache__",
}

# repo-root files the README and notebooks reference by repo-root-relative path.
_ROOT_FILES = ("README.md", "CONTRIBUTING.md", "CITATION.cff", "LICENSE")


def _copy_docs_content(dst: Path) -> None:
    """Copy the real docs/ documentation content into ``dst`` (mirror/docs)."""
    dst.mkdir(parents=True)
    for item in sorted(_DOCS.iterdir()):
        if item.name in _DOCS_MACHINERY:
            continue
        if item.is_dir():
            shutil.copytree(item, dst / item.name)
        else:
            shutil.copy2(item, dst / item.name)


def main() -> None:
    if _SITE.exists():
        shutil.rmtree(_SITE)
    _SITE.mkdir(parents=True)

    for name in _ROOT_FILES:
        shutil.copy2(_REPO / name, _SITE / name)

    # site landing page + API pages authored in docs/, placed at the mirror root
    shutil.copy2(_DOCS / "index.md", _SITE / "index.md")
    if (_DOCS / "api").exists():
        shutil.copytree(_DOCS / "api", _SITE / "api")

    # repo documentation content -> mirror/docs/ (so ../../docs/X and docs/X both resolve)
    _copy_docs_content(_SITE / "docs")

    # the tutorial notebooks -> mirror/tutorials/ (single source of truth stays in-repo)
    shutil.copytree(
        _REPO / "tutorials",
        _SITE / "tutorials",
        ignore=shutil.ignore_patterns(".ipynb_checkpoints", "__pycache__"),
    )


if __name__ == "__main__":
    main()
