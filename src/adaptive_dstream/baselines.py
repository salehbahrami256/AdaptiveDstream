from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np

from .cell import GridCell
from .model import AdaptiveDStream


@dataclass
class FixedGridDStream(AdaptiveDStream):
    """A classic, non-adaptive D-Stream: the domain is partitioned once into
    an ``n_cells_per_dim ** dim`` uniform grid and never re-partitioned.

    Everything else (exponential decay, dense/sparse/transitional states,
    face-adjacency clustering, pruning, prediction) is inherited unchanged
    from :class:`AdaptiveDStream` — a fixed grid is the special case of the
    adaptive model where splitting never fires. This makes it a faithful,
    apples-to-apples baseline rather than a re-implementation with subtly
    different semantics.
    """

    n_cells_per_dim: int = 10

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.n_cells_per_dim < 1:
            raise ValueError("n_cells_per_dim must be >= 1.")
        edges = [
            np.linspace(self.lower[j], self.upper[j], self.n_cells_per_dim + 1)
            for j in range(self.dim)
        ]
        children = []
        for idx in itertools.product(range(self.n_cells_per_dim), repeat=self.dim):
            lo = np.array([edges[j][idx[j]] for j in range(self.dim)])
            hi = np.array([edges[j][idx[j] + 1] for j in range(self.dim)])
            children.append(GridCell(lo, hi, level=1, last_update=self.t))
        self.root.children = children
        # Row-major flat index -> cell, matching itertools.product's
        # iteration order above. Kept as our own reference (not derived from
        # leaves(), whose stack-based traversal order is reversed) so
        # _find_leaf can do O(1) arithmetic indexing instead of a linear
        # containment scan over all n_cells_per_dim ** dim cells.
        self._grid_leaves = children
        self._cell_widths = (self.upper - self.lower) / self.n_cells_per_dim

    def _should_split(self, cell: GridCell) -> bool:
        # A fixed-resolution D-Stream never refines its grid.
        return False

    def _find_leaf(self, x: np.ndarray) -> GridCell:
        x = np.clip(x, self.lower, self.upper)
        idx = np.minimum(
            ((x - self.lower) / self._cell_widths).astype(int), self.n_cells_per_dim - 1
        )
        flat = 0
        for j in range(self.dim):
            flat = flat * self.n_cells_per_dim + int(idx[j])
        return self._grid_leaves[flat]

    def maintenance(self) -> None:
        # A fixed grid's defining property is that its partition never
        # changes. AdaptiveDStream.maintenance() also prunes idle/sparse
        # leaves, which for an adaptively-split tree means reverting a
        # refinement — but here it would delete grid cells outright,
        # leaving spatial gaps and silently breaking the "fixed resolution"
        # contract (a later point in the gap would snap to whatever
        # neighboring cell happens to be nearest). So: reassign clusters,
        # never prune.
        self._assign_clusters()


class RiverClusterAdapter:
    """Wraps a river streaming-clustering model (DenStream / CluStream /
    DBSTREAM, ...) behind the same minimal interface used by the harness for
    :class:`AdaptiveDStream`: ``partial_fit(x, t)``, ``predict_point_cluster(x)``,
    ``summary()``.

    river models take dict-of-features input and mutate in place via
    ``learn_one`` (it does not return ``self``), so this adapter only exists
    to paper over that interface mismatch for :mod:`evaluation`.
    """

    def __init__(self, model, name: str | None = None):
        self.model = model
        self.name = name or type(model).__name__
        self.t = 0

    @staticmethod
    def _to_dict(x: np.ndarray) -> dict:
        return {i: float(v) for i, v in enumerate(np.asarray(x, dtype=float))}

    def partial_fit(self, x: np.ndarray, t: int | None = None) -> "RiverClusterAdapter":
        self.t = t if t is not None else self.t + 1
        self.model.learn_one(self._to_dict(x))
        return self

    def predict_point_cluster(self, x: np.ndarray) -> int | None:
        try:
            label = self.model.predict_one(self._to_dict(x))
        except Exception:
            return None
        return None if label is None else int(label)

    def summary(self) -> dict:
        n_clusters = getattr(self.model, "n_clusters", None)
        if n_clusters is None:
            micro = getattr(self.model, "micro_clusters", None)
            n_clusters = len(micro) if micro is not None else None
        return {"n_leaves": n_clusters, "n_dense_leaves": n_clusters, "n_clusters": n_clusters}


def make_river_baselines(seed: int = 0) -> dict:
    """Construct the three river baselines named in the evaluation plan,
    with mildly tuned defaults for small 2D streams. Import is local so the
    rest of the package works even where river isn't installed.
    """
    from river import cluster

    return {
        "DenStream": RiverClusterAdapter(
            cluster.DenStream(decaying_factor=0.02, beta=0.5, mu=3, epsilon=0.3, n_samples_init=50),
            name="DenStream",
        ),
        "CluStream": RiverClusterAdapter(
            cluster.CluStream(n_macro_clusters=5, max_micro_clusters=50, time_window=500, time_gap=100, seed=seed),
            name="CluStream",
        ),
        "DBSTREAM": RiverClusterAdapter(
            cluster.DBSTREAM(clustering_threshold=0.5, fading_factor=0.01, cleanup_interval=2, intersection_factor=0.3),
            name="DBSTREAM",
        ),
    }
