"""Direct pins on the shared EDOC subprocess discipline (``_subprocess.py``).

These crash-vs-censor branches of
:func:`~tabench.models.adapters._subprocess.run_disciplined` /
:func:`~tabench.models.adapters._subprocess.remaining` are UNREACHABLE
engine-free through the three adapter suites — they need a crashing / hanging /
missing engine — so the shared module is pinned DIRECTLY here (a justified
deviation from the no-dedicated-unit-file convention: after the B3 extraction a
typing drift at this ONE shared point would pass all three adapter suites at
once, so the shared branches earn their own pin):

* the nonzero-rc branch (adr-036 R6): ``censor_on_fail`` decides
  :class:`~tabench.edoc.replay.PlanReplayFailure` (the censor signal — an
  unexecutable emission) vs ``RuntimeError`` (certifier-side infra);
* the TIMEOUT typing split (the S3 review's F1): ``censor_on_timeout`` decides
  the same for a wall-deadline kill INDEPENDENTLY of ``censor_on_fail`` — a
  caller wall clipped below the scenario deadline is budget exhaustion (infra),
  not the plan's fault;
* a missing binary is ALWAYS infra (``RuntimeError`` "could not execute"), never
  a censor, even under ``censor_on_fail=True`` — it is not the plan crashing the
  engine;
* :func:`remaining` pre-exhaustion RAISES with the ROW-LABELLED message, so a
  swapped-label regression across the three rows is caught here.

Everything runs on ``sys.executable`` / a bogus path — no engine, sub-second."""

from __future__ import annotations

import os
import sys
import time

import pytest

from tabench.edoc.replay import PlanReplayFailure
from tabench.models.adapters import _subprocess as sp


def test_nonzero_rc_censor_vs_infra_matrix(tmp_path):
    """A nonzero exit is a PlanReplayFailure censor under ``censor_on_fail=True``
    (the replay step) and a plain RuntimeError otherwise (adr-036 R6); both
    messages carry the exit code and the stderr-tail marker (``rc`` never
    trusted — the failure is reported, not laundered)."""
    cmd = [sys.executable, "-c", "import sys; sys.stderr.write('RCTAIL'); sys.exit(3)"]

    with pytest.raises(PlanReplayFailure) as ei:
        sp.run_disciplined(
            cmd, cwd=str(tmp_path), deadline=None, what="rc-pin",
            env={**os.environ}, label="test", censor_on_fail=True,
        )
    assert "exit 3" in str(ei.value)
    assert "stderr tail:" in str(ei.value) and "RCTAIL" in str(ei.value)

    with pytest.raises(RuntimeError) as ei2:
        sp.run_disciplined(
            cmd, cwd=str(tmp_path), deadline=None, what="rc-pin",
            env={**os.environ}, label="test", censor_on_fail=False,
        )
    assert not isinstance(ei2.value, PlanReplayFailure)  # infra typing, not the censor signal
    assert "exit 3" in str(ei2.value)


def test_wall_timeout_typing_split(tmp_path):
    """The S3 F1 timeout split: a wall-deadline kill censors
    (PlanReplayFailure) only when ``censor_on_timeout`` holds (the default
    follows ``censor_on_fail``); a caller-clipped wall passes
    ``censor_on_timeout=False`` and RAISES infra instead — even with
    ``censor_on_fail=True``. Both messages name the wall-deadline kill."""
    sleeper = [sys.executable, "-c", "import time; time.sleep(30)"]

    with pytest.raises(PlanReplayFailure) as ei:
        sp.run_disciplined(
            sleeper, cwd=str(tmp_path), deadline=time.perf_counter() + 0.6,
            what="to-pin", env={**os.environ}, label="test", censor_on_fail=True,
        )
    assert "killed by the wall deadline" in str(ei.value)

    with pytest.raises(RuntimeError) as ei2:
        sp.run_disciplined(
            sleeper, cwd=str(tmp_path), deadline=time.perf_counter() + 0.6,
            what="to-pin", env={**os.environ}, label="test",
            censor_on_fail=True, censor_on_timeout=False,
        )
    assert not isinstance(ei2.value, PlanReplayFailure)  # caller-clip is budget, not censor
    assert "killed by the wall deadline" in str(ei2.value)


def test_missing_binary_is_always_infra(tmp_path):
    """A missing binary (``OSError`` from ``Popen``) is ALWAYS certifier-side
    infra ("could not execute"), never a censor — even under
    ``censor_on_fail=True`` — because it is not the plan crashing the engine."""
    with pytest.raises(RuntimeError) as ei:
        sp.run_disciplined(
            ["/nonexistent/tabench-no-such-binary"], cwd=str(tmp_path),
            deadline=None, what="missing-pin", env={**os.environ},
            label="test", censor_on_fail=True,
        )
    assert not isinstance(ei.value, PlanReplayFailure)
    assert "could not execute" in str(ei.value)


@pytest.mark.parametrize("label", ["sumo-duaiterate", "matsim", "dtalite-simulation"])
def test_remaining_preexhaustion_label(label):
    """Pre-exhaustion RAISES infra with the EXACT row-labelled message, so a
    swapped-label regression is caught for each of the three real rows; an
    unbudgeted deadline is ``None`` and a live one returns positive seconds."""
    with pytest.raises(RuntimeError) as ei:
        sp.remaining(time.perf_counter() - 1.0, label=label)
    assert not isinstance(ei.value, PlanReplayFailure)
    assert str(ei.value) == f"{label} wall deadline exhausted before the next step"
    assert sp.remaining(None, label=label) is None
    assert sp.remaining(time.perf_counter() + 5.0, label=label) > 0
