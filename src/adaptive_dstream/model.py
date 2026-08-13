from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import itertools
import numpy as np

from .cell import GridCell


@dataclass
class AdaptiveDStream:
    lower: np.ndarray
    upper: np.ndarray
    decay: float = 0.995
    dense_threshold: float = 8.0
    sparse_threshold: float = 1.0
    split_threshold: float = 0.04
    max_depth: int = 6
    max_cells: int = 5000
    maintenance_interval: int = 50
    idle_prune_after: int = 300
    alpha_var: float = 1.0
    alpha_mean: float = 1.0

    def __post_init__(self) -> None:
        self.lower = np.asarray(self.lower, dtype=float)
        self.upper = np.asarray(self.upper, dtype=float)
        if self.lower.shape != self.upper.shape:
            raise ValueError("lower and upper must have the same shape.")
        if not np.all(self.upper > self.lower):
            raise ValueError("Each upper bound must be greater than lower bound.")
        if not (0.0 < self.decay < 1.0):
            raise ValueError("decay must be in (0, 1).")
        self.root = GridCell(self.lower, self.upper, level=0, last_update=0)
        self.t = 0
        self.n_seen = 0

    @property
    def dim(self) -> int:
        return len(self.lower)

    def leaves(self) -> list[GridCell]:
        out, stack = [], [self.root]
        while stack:
            node = stack.pop()
            if node.is_leaf:
                out.append(node)
            else:
                stack.extend(node.children)
        return out

    def _find_leaf(self, x: np.ndarray) -> GridCell:
        if not self.root.contains(x):
            raise ValueError(f"Point {x} outside domain [{self.lower}, {self.upper}].")
        node = self.root
        while not node.is_leaf:
            hits = [c for c in node.children if c.contains(x)]
            if hits:
                node = hits[0]
            else:
                centers = np.array([c.center for c in node.children])
                node = node.children[int(np.argmin(np.linalg.norm(centers - x, axis=1)))]
        return node

    def partial_fit(self, x: np.ndarray, t: int | None = None) -> "AdaptiveDStream":
        x = np.asarray(x, dtype=float)
        if x.shape != self.lower.shape:
            raise ValueError(f"Expected point shape {self.lower.shape}, got {x.shape}.")
        if t is None:
            t = self.t + 1
        if t < self.t:
            raise ValueError("Time must be non-decreasing.")

        self.t = int(t)
        self.n_seen += 1
        leaf = self._find_leaf(x)
        leaf.update(x, self.t, self.decay)

        if self._should_split(leaf):
            self._split(leaf)
        if self.n_seen % self.maintenance_interval == 0:
            self.maintenance()
        return self

    def fit_stream(self, X: np.ndarray) -> "AdaptiveDStream":
        for t, x in enumerate(X, start=1):
            self.partial_fit(x, t=t)
        return self

    def _should_split(self, cell: GridCell) -> bool:
        if not cell.is_leaf or cell.level >= self.max_depth:
            return False
        if cell.raw_count < max(4, 2 * self.dim):
            return False
        projected = len(self.leaves()) - 1 + (2 ** self.dim)
        if projected > self.max_cells:
            return False
        score = cell.refinement_score(self.alpha_var, self.alpha_mean)
        return score > self.split_threshold

    def _split(self, cell: GridCell) -> None:
        """Split every dimension; redistribute historical summaries equally."""
        mids = 0.5 * (cell.lower + cell.upper)
        children = []
        for bits in itertools.product([0, 1], repeat=self.dim):
            bits = np.asarray(bits, dtype=int)
            lower = np.where(bits == 0, cell.lower, mids)
            upper = np.where(bits == 0, mids, cell.upper)
            children.append(GridCell(lower, upper, cell.level + 1, cell.last_update))

        k = len(children)
        for child in children:
            # v0 assumption: historical mass is equally distributed among
            # children and is locally uniform inside each child. This gives
            # child-specific moments consistent with that assumption.
            child.s0 = cell.s0 / k
            mu = child.center
            var = (child.side_lengths ** 2) / 12.0
            child.s1 = child.s0 * mu
            child.s2 = child.s0 * (var + mu * mu)
            child.raw_count = cell.raw_count // k
        cell.children = children

    def maintenance(self) -> None:
        self._prune_recursive(self.root)
        self._assign_clusters()

    def _prune_recursive(self, node: GridCell) -> None:
        if node.is_leaf:
            return
        kept = []
        for child in node.children:
            if not child.is_leaf:
                self._prune_recursive(child)
            if child.is_leaf:
                child.decay_to(self.t, self.decay)
                inactive = (self.t - child.last_update) >= self.idle_prune_after
                sparse = child.s0 < self.sparse_threshold
                if inactive and sparse:
                    continue
            kept.append(child)
        node.children = kept

    def _cells_touch_by_face(self, a: GridCell, b: GridCell, tol: float = 1e-12) -> bool:
        touching_dims = 0
        for j in range(self.dim):
            a0, a1 = a.lower[j], a.upper[j]
            b0, b1 = b.lower[j], b.upper[j]
            boundary_touch = abs(a1 - b0) <= tol or abs(b1 - a0) <= tol
            overlap = min(a1, b1) - max(a0, b0)
            if boundary_touch:
                touching_dims += 1
            elif overlap <= tol:
                return False
        return touching_dims == 1

    def dense_leaves(self) -> list[GridCell]:
        return [
            c for c in self.leaves()
            if c.state(self.t, self.decay, self.dense_threshold, self.sparse_threshold) == "dense"
        ]

    def _assign_clusters(self) -> None:
        leaves = self.leaves()
        for c in leaves:
            c.cluster_id = None

        dense = self.dense_leaves()
        neighbors = [[] for _ in dense]
        for i in range(len(dense)):
            for j in range(i + 1, len(dense)):
                if self._cells_touch_by_face(dense[i], dense[j]):
                    neighbors[i].append(j)
                    neighbors[j].append(i)

        visited = set()
        cluster_id = 0
        for start in range(len(dense)):
            if start in visited:
                continue
            q = deque([start])
            visited.add(start)
            while q:
                i = q.popleft()
                dense[i].cluster_id = cluster_id
                for j in neighbors[i]:
                    if j not in visited:
                        visited.add(j)
                        q.append(j)
            cluster_id += 1

    def cluster_cells(self) -> list[GridCell]:
        self._assign_clusters()
        return [c for c in self.leaves() if c.cluster_id is not None]

    def predict_point_cluster(self, x: np.ndarray) -> int | None:
        self._assign_clusters()
        return self._find_leaf(np.asarray(x, dtype=float)).cluster_id

    def summary(self) -> dict:
        leaves = self.leaves()
        dense = self.dense_leaves()
        clustered = self.cluster_cells()
        ids = {c.cluster_id for c in clustered if c.cluster_id is not None}
        return {
            "n_seen": self.n_seen,
            "time": self.t,
            "n_leaves": len(leaves),
            "n_dense_leaves": len(dense),
            "n_clusters": len(ids),
            "max_level": max((c.level for c in leaves), default=0),
        }
