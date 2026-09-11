"""Central logging setup.

Every module in this package logs through the ``adaptive_dstream`` logger
tree (via :func:`get_logger`), and every ``examples/run_*.py`` script wires
that tree to the console *and* to a timestamped file under ``logs/`` via one
call to :func:`configure_run_logging` at the top of its ``main()``.

Why this exists: this is a research codebase where the point of a run is the
*trail* it leaves — what data was fetched (from where, how much, how long),
which features were kept and why, what the model did structurally (splits,
merges, prunes) and when, and the final numbers — not just the final numbers
in isolation. A saved log file makes a run's results reproducible-by-reading
after the fact, without having to re-run it or scroll back through a
terminal. See CLAUDE.md's "Logging" section for the conventions every module
follows (levels, what belongs at INFO vs DEBUG).
"""
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

LOG_DIR = Path("logs")
_LOGGER_ROOT = "adaptive_dstream"


def get_logger(name: str) -> logging.Logger:
    """Return a logger under the shared ``adaptive_dstream`` tree, e.g.
    ``get_logger("real_data")`` -> ``adaptive_dstream.real_data``. Safe to
    call at import time from any module: it does no handler setup itself, so
    importing a module never has a logging side effect. Nothing is emitted
    anywhere until a driver script calls :func:`configure_run_logging`
    (root ``adaptive_dstream`` logger otherwise has no handlers attached, so
    Python's logging falls back to its silent "handler of last resort" at
    WARNING+ — library modules never need to know whether that's happened).
    """
    return logging.getLogger(f"{_LOGGER_ROOT}.{name}")


def configure_run_logging(
    run_name: str, level: int = logging.INFO, log_dir: Path | str = LOG_DIR,
) -> Path:
    """Attach a console handler and a timestamped file handler to the
    ``adaptive_dstream`` logger tree. Call once, near the top of an
    ``examples/run_*.py`` script's ``main()`` (or its ``if __name__ ==
    "__main__":`` block) — every ``get_logger(...)`` call anywhere in the
    package then reaches both handlers for the rest of the process.

    Returns the log file path, so the caller can also print it up front (the
    file only starts filling in *after* this call, so knowing the name
    before a long run is useful if you want to ``tail -f`` it).
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(exist_ok=True)
    log_path = log_dir / f"{run_name}_{time.strftime('%Y%m%d-%H%M%S')}.log"

    root = logging.getLogger(_LOGGER_ROOT)
    root.setLevel(level)
    root.handlers.clear()
    root.propagate = False

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s", "%H:%M:%S")

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    root.addHandler(console)

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(fmt)
    root.addHandler(file_handler)

    get_logger("logging_utils").info("logging to %s (level=%s)", log_path, logging.getLevelName(level))
    return log_path
