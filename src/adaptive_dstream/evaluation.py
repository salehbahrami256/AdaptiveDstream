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
    ari: float = field(init=False)
    nmi: float = field(init=False)
    purity: float = field(init=False)
    throughput_pts_per_sec: float = field(init=False)
    fraction_unassigned: float = field(init=False)
    ari_by_phase: dict = field(init=False)

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
        }


def run_stream_eval(
    model_factory,
    X: np.ndarray,
    y_true: np.ndarray,
    phase: np.ndarray | None = None,
    name: str = "model",
    snapshot_every: int = 100,
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
    """
    n = len(X)
    y_pred = np.empty(n, dtype=int)
    active_cells_over_time: list[tuple[int, int | None]] = []

    tracemalloc.start()
    tracemalloc.reset_peak()
    t0 = time.perf_counter()
    model = model_factory()
    for i in range(n):
        x = X[i]
        pred = model.predict_point_cluster(x)
        y_pred[i] = UNASSIGNED_LABEL if pred is None else pred
        model.partial_fit(x, t=i + 1)
        if (i + 1) % snapshot_every == 0 or i == n - 1:
            active_cells_over_time.append((i + 1, _active_cell_count(model)))
    elapsed = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return EvalResult(
        name=name,
        y_true=np.asarray(y_true),
        y_pred=y_pred,
        phase=None if phase is None else np.asarray(phase),
        active_cells_over_time=active_cells_over_time,
        peak_memory_bytes=peak,
        elapsed_seconds=elapsed,
        n_points=n,
    )
