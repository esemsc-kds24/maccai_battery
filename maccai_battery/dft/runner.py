# =============================================================================
# maccai_battery.dft.runner — Quantum ESPRESSO pw.x job launcher
# =============================================================================
# Provides a single entry-point, run_qe_pw(), for launching QE pw.x
# calculations from Python.
#
# Design goals
# ------------
#   • Streaming  — each line of pw.x output is forwarded to the Python logger
#                  at DEBUG level in real time (no waiting for job completion).
#   • Persistence — all output is written to <run_dir>/qe.out for later parsing
#                   and archival.
#   • Robustness  — per-structure failures are captured in QERunResult.error;
#                   the caller decides whether to abort or continue.
#   • Root safety — OMPI_ALLOW_RUN_AS_ROOT env vars are set automatically,
#                   which is required when running mpirun as root on GPU servers
#                   (e.g. Colab, SLURM jobs started as root).
#   • Timeout     — an optional per-job wall-time limit kills runaway
#                   calculations without aborting the whole pipeline.
#
# Usage
# -----
#   from maccai_battery.dft.runner import run_qe_pw
#   result = run_qe_pw(run_dir=Path("output/dft/scf/scf_0"), nproc=4)
#   if result.success:
#       print("Energy run successful in", result.wall_time_s, "s")
#   else:
#       print("QE failed:", result.error)
# =============================================================================

from __future__ import annotations

import logging
import os
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class QERunResult:
    """Structured result from a single pw.x invocation.

    Attributes
    ----------
    run_dir : Path
        The directory in which pw.x was executed and where ``qe.out`` lives.
    returncode : int
        Process return code.  ``0`` indicates a clean exit; non-zero (or
        ``-1`` for timeout) indicates failure.
    stdout : str
        Combined stdout + stderr from the pw.x process.  Also written to
        ``<run_dir>/qe.out``.
    wall_time_s : float
        Elapsed wall time in seconds from process start to completion
        (or kill, in the case of timeout).
    error : str or None
        Human-readable error description, or ``None`` if the run completed
        without error.  Note that ``error is None`` does not guarantee SCF
        convergence — always check the output for ``"convergence has been
        achieved"``.
    """

    run_dir: Path
    returncode: int
    stdout: str
    wall_time_s: float
    error: Optional[str] = None

    @property
    def success(self) -> bool:
        """``True`` if the process exited cleanly (returncode == 0) and no
        error was recorded.

        Note
        ----
        A ``success=True`` result means QE finished without crashing.
        It does **not** mean the SCF converged — always check the stdout
        for ``"convergence has been achieved"`` or use
        :func:`~maccai_battery.utils.parse_qe_scf_converged`.
        """
        return self.returncode == 0 and self.error is None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _stream_output(
    process: subprocess.Popen,
    lines: list,
    out_path: Path,
) -> None:
    """Read lines from *process.stdout*, log each one, write to *out_path*.

    Intended to run in a background :class:`threading.Thread` so that the
    main thread can call :meth:`subprocess.Popen.wait` with a timeout while
    output is still being streamed.

    Parameters
    ----------
    process : subprocess.Popen
        Running process whose ``stdout`` pipe will be read until EOF.
    lines : list
        Mutable list to which each output line (including newline) is appended.
        The caller collects this into a single string after the thread joins.
    out_path : Path
        Destination file for the streamed output.  Written incrementally so
        that partial output is visible even if the process is later killed.
    """
    try:
        with out_path.open("w", encoding="utf-8", errors="replace") as fh:
            for raw_line in process.stdout:  # type: ignore[union-attr]
                fh.write(raw_line)
                fh.flush()
                lines.append(raw_line)
                logger.debug("[QE] %s", raw_line.rstrip())
    except Exception as exc:  # pragma: no cover
        logger.warning("QE output streamer raised an exception: %s", exc)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_qe_pw(
    run_dir: Path,
    nproc: int = 1,
    executable: str = "pw.x",
    timeout: Optional[int] = None,
) -> QERunResult:
    """Run pw.x in *run_dir*, streaming output to the logger and ``qe.out``.

    The function expects a ``qe.in`` file to be present inside *run_dir*
    (written by :func:`~maccai_battery.dft.input_generator.write_qe_input`).

    Parameters
    ----------
    run_dir : Path
        Directory containing ``qe.in``.  pw.x is executed with this as the
        current working directory so that relative paths in the input file
        (e.g. ``outdir="./out"``) resolve correctly.
    nproc : int, optional
        Number of MPI processes.  When ``nproc == 1`` mpirun is skipped and
        pw.x is invoked directly to avoid MPI overhead on single-core runs.
        When ``nproc > 1``, ``mpirun -np <nproc> pw.x -in qe.in`` is used.
    executable : str, optional
        Name or absolute path of the pw.x binary.  Defaults to ``"pw.x"``
        (resolved via PATH).
    timeout : int or None, optional
        Maximum allowed wall time in seconds.  If the process has not
        finished within *timeout* seconds it is killed with ``SIGKILL`` and
        the returned :class:`QERunResult` will have ``returncode=-1`` and a
        non-``None`` ``error`` field.  ``None`` (default) disables the
        timeout.

    Returns
    -------
    QERunResult
        Contains the process stdout, return code, wall time, and any error
        description.  Never raises on QE-internal failures — those are
        encoded in the return value.

    Raises
    ------
    FileNotFoundError
        If ``pw.x`` or (when ``nproc > 1``) ``mpirun`` cannot be found on
        PATH or at the given path.
    OSError
        If the ``run_dir`` does not exist or is not a directory.

    Notes
    -----
    The following environment variables are injected unconditionally so that
    mpirun works when the process is running as root (e.g. Docker, Colab,
    some HPC job schedulers)::

        OMPI_ALLOW_RUN_AS_ROOT=1
        OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1

    These variables are safe to set on non-root processes and have no effect
    outside OpenMPI.

    Examples
    --------
    >>> from pathlib import Path
    >>> result = run_qe_pw(Path("output/dft/scf/scf_0"), nproc=1)
    >>> print(result.success, result.wall_time_s)
    True 42.7
    """
    run_dir = Path(run_dir)

    # -- Validate run directory ------------------------------------------
    if not run_dir.is_dir():
        raise OSError(
            f"run_dir does not exist or is not a directory: {run_dir}\n"
            "Create it and write qe.in before calling run_qe_pw()."
        )

    qe_in = run_dir / "qe.in"
    if not qe_in.exists():
        logger.warning(
            "qe.in not found in %s — QE will likely fail immediately.", run_dir
        )

    qe_out = run_dir / "qe.out"

    # -- Build command ----------------------------------------------------
    if nproc > 1:
        cmd = [
            "mpirun",
            "-np", str(nproc),
            executable,
            "-in", "qe.in",
        ]
    else:
        # Skip mpirun for single-process runs to avoid MPI overhead and to
        # work on machines where mpirun is not installed.
        cmd = [executable, "-in", "qe.in"]

    # -- Environment: allow mpirun as root --------------------------------
    env = os.environ.copy()
    env["OMPI_ALLOW_RUN_AS_ROOT"] = "1"
    env["OMPI_ALLOW_RUN_AS_ROOT_CONFIRM"] = "1"

    logger.info(
        "Launching QE: cmd=%s  cwd=%s  nproc=%d  timeout=%s",
        " ".join(cmd),
        run_dir,
        nproc,
        f"{timeout}s" if timeout else "none",
    )

    # -- Launch -----------------------------------------------------------
    t0 = time.monotonic()
    try:
        process = subprocess.Popen(
            cmd,
            cwd=run_dir,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,   # merge stderr into stdout
            text=True,
            bufsize=1,                  # line-buffered for real-time streaming
        )
    except FileNotFoundError as exc:
        # pw.x or mpirun not found on PATH
        missing = cmd[0]
        raise FileNotFoundError(
            f"'{missing}' not found. "
            f"{'Install Quantum ESPRESSO and ensure pw.x is on PATH.' if missing == executable else 'Ensure mpirun (OpenMPI) is installed or set nproc=1.'}"
        ) from exc

    # -- Stream output in background thread ------------------------------
    output_lines: list[str] = []
    reader_thread = threading.Thread(
        target=_stream_output,
        args=(process, output_lines, qe_out),
        daemon=True,
        name=f"qe-reader-{run_dir.name}",
    )
    reader_thread.start()

    # -- Wait (with optional timeout) ------------------------------------
    timed_out = False
    try:
        if timeout is not None:
            process.wait(timeout=timeout)
        else:
            process.wait()
    except subprocess.TimeoutExpired:
        timed_out = True
        logger.warning(
            "QE timed out after %ds in %s — killing process.", timeout, run_dir
        )
        process.kill()
        process.wait()

    # Wait for the reader thread to drain any remaining output
    reader_thread.join(timeout=10)

    wall_time_s = time.monotonic() - t0
    stdout = "".join(output_lines)

    # -- Build result -----------------------------------------------------
    if timed_out:
        error_msg = (
            f"pw.x timed out after {timeout}s "
            f"(wall time: {wall_time_s:.1f}s)"
        )
        logger.error("QE TIMEOUT in %s: %s", run_dir, error_msg)
        return QERunResult(
            run_dir=run_dir,
            returncode=-1,
            stdout=stdout,
            wall_time_s=wall_time_s,
            error=error_msg,
        )

    rc = process.returncode

    if rc != 0:
        # Capture last few lines for a concise error message
        last_lines = [ln.strip() for ln in output_lines[-20:] if ln.strip()]
        snippet = " | ".join(last_lines[-5:]) if last_lines else "(no output)"
        error_msg = (
            f"pw.x exited with returncode={rc}. "
            f"Last output: {snippet}"
        )
        logger.error(
            "QE FAILED in %s: returncode=%d, wall_time=%.1fs",
            run_dir,
            rc,
            wall_time_s,
        )
        return QERunResult(
            run_dir=run_dir,
            returncode=rc,
            stdout=stdout,
            wall_time_s=wall_time_s,
            error=error_msg,
        )

    logger.info(
        "QE completed successfully in %s: %.1fs, %d output lines",
        run_dir,
        wall_time_s,
        len(output_lines),
    )
    return QERunResult(
        run_dir=run_dir,
        returncode=rc,
        stdout=stdout,
        wall_time_s=wall_time_s,
        error=None,
    )
