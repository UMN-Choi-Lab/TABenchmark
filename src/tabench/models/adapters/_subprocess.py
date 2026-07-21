"""The shared EDOC subprocess discipline — the wall-deadline / process-group
kill / crash-vs-censor plumbing every external-engine EDOC row runs its engine
calls under (adr-036 R6; the per-row applications in adr-037 sumo / adr-039
matsim / adr-040 dtalite, with the underlying anti-stale-toolchain hazard
doctrine in adr-027 sumo / adr-029 dtalite).

This is the SHARED HOME for the "S2 discipline" the three EDOC producer adapters
(``sumo_duaiterate``, ``matsim_edoc``, ``dtalite_simulation``) each carried
verbatim, so a fix to the timeout / kill / censor semantics lands ONCE here
instead of drifting across three copies. Each adapter binds the two
engine-specific dials — the subprocess ``env`` and the wall-deadline message
``label`` (its row name) — in a thin local ``_run`` wrapper and otherwise calls
straight through; the shape (a list ``cmd``, the S3 ``censor_on_timeout`` split)
is identical.

**The discipline (measured hazards, never laundered):**

* a SINGLE wall deadline threads every engine call of one certify; :func:`remaining`
  returns the seconds left and RAISES ``RuntimeError`` (infra) if a prior phase
  already ate the budget — a pre-exhaustion is never a censor;
* ``stdin=DEVNULL`` and ``start_new_session=True`` so each call is its OWN
  process group: a wall-deadline kill ``killpg(SIGKILL)``s the whole tree, not
  just the direct child (:func:`reap_group`). A SUMO tool spawns sumo/duarouter
  grandchildren, a JVM spawns ``jspawnhelper`` children, and DTALite ctypes-loads
  an OpenMP engine into the python child; all orphan to init and keep burning
  CPU/RAM if only the direct child is killed (F2, measured on every engine);
* the crash-vs-censor typing (adr-036 R6, the S3 F1 split): by DEFAULT a timeout
  / OS error / nonzero rc is a certifier-side INFRASTRUCTURE ``RuntimeError``;
  the ONE step that replays the MODEL's emitted plans passes
  ``censor_on_fail=True`` so a genuine engine crash raises
  :class:`~tabench.edoc.replay.PlanReplayFailure` (the censor signal — an
  unexecutable / head-blocking plan is an invalid emission). The TIMEOUT typing
  is split out separately: ``censor_on_timeout`` (default: follows
  ``censor_on_fail``) is passed ``False`` when a CALLER wall clipped below the
  scenario-declared deadline, so a certifier-side budget kill RAISES instead of
  censoring — only a scenario-deadline expiry blames the plan;
* ``rc`` is NEVER trusted: :func:`run_disciplined` only guarantees a clean exit;
  every caller re-reads the artifact the step was supposed to write.

:func:`intersect_replay_deadline` derives the certifier's hard replay deadline
from the scenario's hashed ``replay_deadline_s`` (intersected with any tighter
caller wall) and returns the ``clipped_by_caller`` flag that drives that timeout
typing.

Pure stdlib + :mod:`tabench.edoc` — no engine import at module scope, so this
imports everywhere the numpy/scipy core does (the adapters' unconditional-import
posture). Design: docs/design/adr-036 + adr-037 / adr-039 / adr-040.
"""

from __future__ import annotations

import os
import signal
import subprocess
import time
from typing import TYPE_CHECKING

from ...edoc.replay import PlanReplayFailure

if TYPE_CHECKING:
    from ...edoc.scenario import EdocScenario


def remaining(deadline: float | None, *, label: str) -> float | None:
    """Seconds left on the single wall deadline, or ``None`` if unbudgeted.
    RAISES ``RuntimeError`` (infra) if the deadline already passed (a prior phase
    ate the whole budget). ``label`` is the row name, so the message names the
    engine whose budget expired."""
    if deadline is None:
        return None
    left = deadline - time.perf_counter()
    if left <= 0.0:
        raise RuntimeError(f"{label} wall deadline exhausted before the next step")
    return left


def reap_group(proc: subprocess.Popen) -> None:
    """SIGKILL the subprocess's whole process GROUP, then reap it (F2).
    ``subprocess`` times out only the direct child, but an engine call spawns a
    tree — SUMO tools spawn sumo/duarouter grandchildren, a JVM spawns
    ``jspawnhelper`` children, DTALite ctypes-loads an OpenMP engine into the
    python child — that otherwise orphans to init and keeps burning CPU/RAM after
    the wall fires (measured). ``start_new_session=True`` puts the child in its
    own group so one ``killpg`` takes the whole tree down."""
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        proc.kill()  # group already gone / not the group leader: kill the child
    try:
        proc.wait(timeout=10)
    except (subprocess.TimeoutExpired, OSError):
        pass


def intersect_replay_deadline(
    scenario: EdocScenario, deadline: float | None
) -> tuple[float, bool]:
    """The certifier's hard replay deadline (F3): the scenario-declared hashed
    ``replay_deadline_s`` measured from now, intersected with any tighter caller
    wall — so the hashed constant ALWAYS bounds a certifier replay and a
    head-blocking plan cannot hang the certifier unboundedly (adr-036 R6).

    Returns ``(deadline, clipped_by_caller)``. The flag carries the R6
    crash-vs-censor typing for a mid-replay timeout (the S3 F1 split): only an
    expiry of the SCENARIO-declared deadline is the model's fault (an
    unexecutable / head-blocking plan — censor); a caller wall clipping below it
    is a certifier-side budget exhaustion and must RAISE as infrastructure,
    never censor."""
    scen_deadline = time.perf_counter() + float(scenario.replay_deadline_s)
    if deadline is None or deadline >= scen_deadline:
        return scen_deadline, False
    return deadline, True


def run_disciplined(
    cmd: list[str],
    *,
    cwd: str,
    deadline: float | None,
    what: str,
    env: dict[str, str],
    label: str,
    censor_on_fail: bool = False,
    censor_on_timeout: bool | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run one subprocess under the shared EDOC discipline: the passed ``env``,
    ``stdin=DEVNULL``, output captured, its OWN process group so a wall kill
    reaps the whole tree (F2), the single wall deadline (via :func:`remaining`,
    ``label``-tagged) as timeout.

    Crash-vs-censor (adr-036 R6, S3 F1 split): by default a timeout / OS error /
    nonzero rc is a certifier-side INFRASTRUCTURE ``RuntimeError``. For the ONE
    step that replays the MODEL's emitted plans, the caller passes
    ``censor_on_fail=True`` so a genuine subprocess crash raises
    :class:`~tabench.edoc.replay.PlanReplayFailure` (the censor signal). The
    TIMEOUT typing is split out: ``censor_on_timeout`` (default: follows
    ``censor_on_fail``) is passed ``False`` when a CALLER wall clipped below the
    scenario-declared deadline, so a certifier-side budget kill RAISES instead of
    censoring — only a scenario-deadline expiry blames the plan. A
    :func:`remaining` pre-exhaustion and a missing binary (``OSError``) stay
    infrastructure RAISEs on every step, replay included. ``rc`` is never trusted
    beyond a clean exit — every caller re-reads its artifact."""
    if censor_on_timeout is None:
        censor_on_timeout = censor_on_fail
    timeout = remaining(deadline, label=label)  # pre-exhaustion -> plain RuntimeError (infra)
    try:
        proc = subprocess.Popen(
            cmd,
            cwd=cwd,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
    except OSError as exc:
        raise RuntimeError(f"{what}: could not execute ({exc})\n  cmd: {' '.join(cmd)}") from exc
    try:
        out, err = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        reap_group(proc)
        msg = f"{what}: killed by the wall deadline\n  cmd: {' '.join(cmd)}"
        if censor_on_timeout:
            raise PlanReplayFailure(msg) from exc
        raise RuntimeError(msg) from exc
    if proc.returncode != 0:
        msg = (
            f"{what}: exit {proc.returncode}\n  cmd: {' '.join(cmd)}\n"
            f"  stderr tail: {err[-800:]}"
        )
        if censor_on_fail:
            raise PlanReplayFailure(msg)
        raise RuntimeError(msg)
    return subprocess.CompletedProcess(cmd, proc.returncode, out, err)
