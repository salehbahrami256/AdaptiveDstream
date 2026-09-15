"""Evaluation harness for streaming-clustering models on labeled synthetic
streams.

Unassigned-point convention
----------------------------
Grid/density-based models (``AdaptiveDStream``, ``FixedGridDStream``) return
``None`` from ``predict_point_cluster`` for points that fall outside any
dense cell. ARI/NMI/purity are undefined against a ``None`` label, so this
module maps every unassigned prediction to the sentinel label ``-1`` and
treats it as *its own* predicted cluster (never merged with, or ignored
against, a true label). This mirrors the standard convention for
density-based clustering (e.g. DBSCAN/DenStream "noise"): a model that
dumps everything into "unassigned" is scored as if it put every such point
in one giant extra cluster, which is exactly the behavior ARI/NMI already
penalize as a poor partition. It is a deliberate choice, not an artifact —
without it, ARI/NMI/purity are not comparable across models that do and do
not expose an explicit noise/outlier state.
"""
from __future__ import annotations

import time
import tracemalloc
from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from .logging_utils import get_logger

log = get_logger("evaluation")

UNASSIGNED_LABEL = -1


def purity_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Fraction of points whose predicted cluster's majority true label
    matches their own true label. Not implemented in scikit-learn.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if len(y_true) == 0:
        return float("nan")
    total_correct = 0
    for cluster in np.unique(y_pred):
        mask = y_pred == cluster
        labels, counts = np.unique(y_true[mask], return_counts=True)
        total_correct += counts.max()
    return total_correct / len(y_true)


def decayed_purity_score(y_true: np.ndarray, y_pred: np.ndarray, decay: float) -> float:
    """Recency-weighted purity: the same majority-label-per-cluster idea as
    :func:`purity_score`, but each point ``i`` (arrival order, 0-indexed) is
    weighted ``decay ** (n - 1 - i)`` — the most recent point gets weight 1,
    older points fade out at exactly the rate ``AdaptiveDStream``'s own
    ``GridCell.decay_to`` uses internally.

    This exists because the ordinary (flat) ARI/NMI/purity above weight
    every point in the stream equally regardless of arrival time, which
    silently assumes "every timestep matters forever" — an assumption none
    of the decay-based models being compared (``AdaptiveDStream``,
    ``FixedGridDStream``, river's ``DenStream``/``DBSTREAM``) actually make
    internally, and one that ``CluStream`` (windowed, not decayed) doesn't
    make either. A flat aggregate is not a neutral yardstick; this is a
    complementary, explicitly recency-weighted one, not a replacement.

    Purity generalizes cleanly to a weighted version (sum per-cluster
    weighted majority-label mass, divide by total weight) because it is a
    per-point statistic. ARI is not — it's defined via combinatorial counts
    over *pairs* of points, and its "adjusted for chance" correction assumes
    a specific null distribution over contingency tables that changes under
    weights, so a decay-weighted ARI needs a properly re-derived correction
    term rather than a naive weighted plug-in. Not attempted here; see
    ``run_stream_eval``'s ``eval_decay`` docstring for the windowed
    ARI/NMI used instead, and ``research_notes.txt`` if this gets revisited.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)
    if n == 0:
        return float("nan")
    weights = decay ** (n - 1 - np.arange(n))
    total_correct = 0.0
    for cluster in np.unique(y_pred):
        mask = y_pred == cluster
        labels = np.unique(y_true[mask])
        best = max(weights[mask][y_true[mask] == label].sum() for label in labels)
        total_correct += best
    return float(total_correct / weights.sum())


def _active_cell_count(model) -> int | None:
    summary_fn = getattr(model, "summary", None)
    if summary_fn is None:
        return None
    s = summary_fn()
    return s.get("n_leaves", s.get("n_clusters"))


@dataclass
class EvalResult:
    name: str
    y_true: np.ndarray
    y_pred: np.ndarray
    phase: np.ndarray | None
    active_cells_over_time: list[tuple[int, int | None]]
    peak_memory_bytes: int
    elapsed_seconds: float
    n_points: int
    eval_decay: float | None = None
    ari: float = field(init=False)
    nmi: float = field(init=False)
    purity: float = field(init=False)
    throughput_pts_per_sec: float = field(init=False)
    fraction_unassigned: float = field(init=False)
    ari_by_phase: dict = field(init=False)
    decayed_purity: float | None = field(init=False)
    ari_recent: float | None = field(init=False)
    nmi_recent: float | None = field(init=False)
    recent_window: int | None = field(init=False)

    def __post_init__(self) -> None:
        self.ari = adjusted_rand_score(self.y_true, self.y_pred)
        self.nmi = normalized_mutual_info_score(self.y_true, self.y_pred)
        self.purity = purity_score(self.y_true, self.y_pred)
        self.throughput_pts_per_sec = self.n_points / self.elapsed_seconds if self.elapsed_seconds > 0 else float("inf")
        self.fraction_unassigned = float(np.mean(self.y_pred == UNASSIGNED_LABEL))
        self.ari_by_phase = {}
        if self.phase is not None:
            for p in np.unique(self.phase):
                mask = self.phase == p
                if mask.sum() > 1:
                    self.ari_by_phase[int(p)] = adjusted_rand_score(self.y_true[mask], self.y_pred[mask])

        self.decayed_purity = None
        self.ari_recent = None
        self.nmi_recent = None
        self.recent_window = None
        if self.eval_decay is not None:
            self.decayed_purity = decayed_purity_score(self.y_true, self.y_pred, self.eval_decay)
            # Effective sample size of an exponential fade at this decay
            # rate (see research_notes.txt sec. 8): 1/(1-decay) points carry
            # most of the weight. Recomputing plain, unmodified ARI/NMI over
            # just that many of the most recent points gives a genuinely
            # recency-focused reading using trusted sklearn math, sidestepping
            # the need to re-derive ARI's chance-correction under weights
            # (see decayed_purity_score's docstring for why that's not done
            # here). This window is fixed externally by eval_decay, applied
            # identically to every model in a sweep -- it does not read any
            # model's own internal decay/fading parameter, so it stays a
            # single, controlled comparison lens across models whose own
            # rates differ (or, for CluStream, don't exist at all).
            w = min(self.n_points, int(np.ceil(1.0 / (1.0 - self.eval_decay))))
            self.recent_window = w
            if w > 1:
                self.ari_recent = adjusted_rand_score(self.y_true[-w:], self.y_pred[-w:])
                self.nmi_recent = normalized_mutual_info_score(self.y_true[-w:], self.y_pred[-w:])

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "n_points": self.n_points,
            "ari": self.ari,
            "nmi": self.nmi,
            "purity": self.purity,
            "fraction_unassigned": self.fraction_unassigned,
            "peak_memory_bytes": self.peak_memory_bytes,
            "elapsed_seconds": self.elapsed_seconds,
            "throughput_pts_per_sec": self.throughput_pts_per_sec,
            "ari_by_phase": self.ari_by_phase,
            "active_cells_over_time": self.active_cells_over_time,
            "eval_decay": self.eval_decay,
            "decayed_purity": self.decayed_purity,
            "ari_recent": self.ari_recent,
            "nmi_recent": self.nmi_recent,
            "recent_window": self.recent_window,
        }


def run_stream_eval(
    model_factory,
    X: np.ndarray,
    y_true: np.ndarray,
    phase: np.ndarray | None = None,
    name: str = "model",
    snapshot_every: int = 100,
    eval_decay: float | None = None,
) -> EvalResult:
    """Build a fresh model via ``model_factory()`` and drive it over the
    stream ``X``, scoring it against ``y_true``.

    ``model_factory`` is a zero-argument callable rather than a pre-built
    model for two reasons: it guarantees each run starts from a clean state
    (no leaked state between sweep entries), and it lets peak-memory
    tracking include the cost of allocating the model's initial structure
    (e.g. a fixed grid's ``n_cells_per_dim ** dim`` cells) — which for
    grid-based methods is the dominant memory cost, not something that
    should be hidden by measuring only the online-update phase.

    Uses the standard prequential ("test-then-train") protocol: at each step
    the model predicts the incoming point's cluster *before* it is allowed
    to update on it, so accuracy reflects genuinely online performance
    rather than a hindsight fit. Peak memory is sampled via ``tracemalloc``
    across model construction and the whole run (captures Python + NumPy
    allocations attributable to the model, not baseline interpreter/import
    memory, which is untracked because tracing starts after imports).

    Note: ``tracemalloc`` tracing adds overhead to every allocation, so
    absolute throughput numbers are lower than an un-instrumented run would
    give. That overhead is applied identically to every model in a sweep,
    so relative throughput comparisons between models remain valid.

    ``eval_decay`` (default ``None``, i.e. off — every field below stays
    ``None`` and existing callers/results are unaffected) adds a second,
    explicitly recency-weighted view alongside the ordinary (flat, every
    point equally weighted) ARI/NMI/purity above: ``decayed_purity`` and a
    windowed ``ari_recent``/``nmi_recent`` over the most recent
    ``recent_window = ceil(1/(1-eval_decay))`` points (that formula is the
    effective sample size of an exponential fade at this rate — see
    ``research_notes.txt`` sec. 8). This rate is a single value fixed by the
    caller and applied identically to every model in a sweep; it is
    *not* read from any individual model's own internal decay/fading
    parameter (``AdaptiveDStream.decay``, river's ``decaying_factor``/
    ``fading_factor``, or CluStream's windowing, which don't share a
    timescale in the first place), so this stays one controlled comparison
    lens, not a different one per model. See ``decayed_purity_score``'s
    docstring for why ARI itself isn't decay-weighted directly.
    """
    n = len(X)
    y_pred = np.empty(n, dtype=int)
    active_cells_over_time: list[tuple[int, int | None]] = []

    log.info("run_stream_eval start: name=%r n_points=%d dim=%d", name, n, X.shape[1])
    tracemalloc.start()
    tracemalloc.reset_peak()
    t0 = time.perf_counter()
    model = model_factory()
    progress_every = max(1, n // 4)  # log at ~25/50/75/100% so long real-data runs stay followable
    for i in range(n):
        x = X[i]
        pred = model.predict_point_cluster(x)
        y_pred[i] = UNASSIGNED_LABEL if pred is None else pred
        model.partial_fit(x, t=i + 1)
        if (i + 1) % snapshot_every == 0 or i == n - 1:
            active_cells_over_time.append((i + 1, _active_cell_count(model)))
        if (i + 1) % progress_every == 0 or i == n - 1:
            log.info("run_stream_eval %r progress: %d/%d points (%.0f%%), %.1fs elapsed",
                      name, i + 1, n, 100 * (i + 1) / n, time.perf_counter() - t0)
    elapsed = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    result = EvalResult(
        name=name,
        y_true=np.asarray(y_true),
        y_pred=y_pred,
        phase=None if phase is None else np.asarray(phase),
        active_cells_over_time=active_cells_over_time,
        eval_decay=eval_decay,
        peak_memory_bytes=peak,
        elapsed_seconds=elapsed,
        n_points=n,
    )
    log.info("run_stream_eval done: name=%r ari=%.3f nmi=%.3f purity=%.3f fraction_unassigned=%.2f "
             "peak_memory=%.1fKB elapsed=%.1fs throughput=%.0fpts/s",
             name, result.ari, result.nmi, result.purity, result.fraction_unassigned,
             peak / 1024, elapsed, result.throughput_pts_per_sec)
    return result
