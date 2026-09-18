from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import itertools
import numpy as np
from scipy.stats import norm

from .cell import GridCell
from .logging_utils import get_logger

log = get_logger("model")

SPLIT_STRATEGIES = ("equal_uniform", "point_mass", "moment_based")


def _axis_lower_fraction(mu: float, sigma: float, lo: float, mid: float, hi: float) -> float:
    """P(X <= mid | lo <= X <= hi) for X ~ Normal(mu, sigma^2).

    Degenerates gracefully to a point mass at ``mu`` (clipped into
    ``[lo, hi]``) when ``sigma`` is effectively zero, and to an even 0.5
    split when the parent's own tracked density gives no usable signal
    (``sigma`` non-degenerate but the truncated mass is numerically zero).
    """
    if hi - lo <= 1e-12:
        return 0.5
    if sigma <= 1e-9:
        mu_clipped = min(max(mu, lo), hi)
        return 1.0 if mu_clipped <= mid else 0.0
    alpha, m, beta = (lo - mu) / sigma, (mid - mu) / sigma, (hi - mu) / sigma
    z = norm.cdf(beta) - norm.cdf(alpha)
    if z <= 1e-12:
        return 0.5
    return float((norm.cdf(m) - norm.cdf(alpha)) / z)


def _truncated_normal_moments(mu: float, sigma: float, a: float, b: float) -> tuple[float, float]:
    """Mean and variance of Normal(mu, sigma^2) truncated to [a, b]."""
    if b - a <= 1e-12:
        return 0.5 * (a + b), 0.0
    if sigma <= 1e-9:
        return min(max(mu, a), b), 0.0
    alpha, beta = (a - mu) / sigma, (b - mu) / sigma
    z = norm.cdf(beta) - norm.cdf(alpha)
    if z <= 1e-12:
        return 0.5 * (a + b), ((b - a) ** 2) / 12.0
    phi_a, phi_b = norm.pdf(alpha), norm.pdf(beta)
    mean = mu + sigma * (phi_a - phi_b) / z
    var = (sigma ** 2) * (1.0 + (alpha * phi_a - beta * phi_b) / z - ((phi_a - phi_b) / z) ** 2)
    return mean, max(var, 0.0)


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
    split_strategy: str = "equal_uniform"
    merge_threshold: float = 0.01
    merge_min_age: int = 100
    density_normalize: bool = False

    def __post_init__(self) -> None:
        self.lower = np.asarray(self.lower, dtype=float)
        self.upper = np.asarray(self.upper, dtype=float)
        if self.lower.shape != self.upper.shape:
            raise ValueError("lower and upper must have the same shape.")
        if not np.all(self.upper > self.lower):
            raise ValueError("Each upper bound must be greater than lower bound.")
        if not (0.0 < self.decay < 1.0):
            raise ValueError("decay must be in (0, 1).")
        if self.split_strategy not in SPLIT_STRATEGIES:
            raise ValueError(f"split_strategy must be one of {SPLIT_STRATEGIES}, got {self.split_strategy!r}.")
        if not (self.merge_threshold < self.split_threshold):
            raise ValueError("merge_threshold must be < split_threshold to avoid split/merge oscillation.")
        self.root = GridCell(self.lower, self.upper, level=0, last_update=0)
        self.t = 0
        self.n_seen = 0
        log.info("constructed %s(dim=%d, dense_threshold=%s, sparse_threshold=%s, "
                 "split_threshold=%s, max_depth=%s, max_cells=%s, split_strategy=%r, decay=%s, "
                 "density_normalize=%s)",
                 type(self).__name__, self.dim, self.dense_threshold, self.sparse_threshold,
                 self.split_threshold, self.max_depth, self.max_cells, self.split_strategy, self.decay,
                 self.density_normalize)

    @property
    def dim(self) -> int:
        return len(self.lower)

    @property
    def domain_volume(self) -> float:
        return float(np.prod(self.upper - self.lower))

    def _volume_scale(self, cell: GridCell) -> float:
        """See ``density_normalize`` and ``GridCell.state`` for the full
        rationale. Off (1.0, i.e. compare raw decayed count) by default to
        keep existing calibrated thresholds' behavior unchanged; when
        enabled, rescales a cell's count to what it would be at the root's
        volume, so ``dense_threshold``/``sparse_threshold`` are compared
        against a density rather than an absolute count that silently means
        less every time a region gets refined."""
        if not self.density_normalize:
            return 1.0
        return self.domain_volume / max(cell.volume, 1e-300)

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
        # GridCell.contains is half-open ([lower, upper) per axis) so that a
        # point on a face shared by two sibling cells matches exactly one of
        # them. That leaves exactly one point unrepresented by any leaf: the
        # global upper corner of the whole domain itself (x == self.upper on
        # some axis). Nudge the *search key* used to descend the tree
        # infinitesimally below self.upper so it always falls inside the
        # half-open rightmost cell; this never touches the actual point used
        # to update statistics (x_clipped in partial_fit), only which leaf
        # receives it.
        search_key = np.clip(x, self.lower, self.upper)
        search_key = np.where(
            search_key >= self.upper, np.nextafter(self.upper, self.lower), search_key,
        )
        node = self.root
        while not node.is_leaf:
            hits = [c for c in node.children if c.contains(search_key)]
            if hits:
                node = hits[0]
            else:
                # Should be unreachable given the nudge above; if it fires,
                # it means floating-point error accumulated across repeated
                # splits pushed search_key outside every child's bounds.
                # Surface it instead of silently guessing.
                log.warning(
                    "t=%d _find_leaf: no child of level=%d center=%s contains search_key=%s "
                    "(falling back to nearest child centre -- investigate float precision)",
                    self.t, node.level, np.round(node.center, 6), np.round(search_key, 6),
                )
                centers = np.array([c.center for c in node.children])
                node = node.children[int(np.argmin(np.linalg.norm(centers - search_key, axis=1)))]
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
        x_clipped = np.clip(x, self.lower, self.upper)
        leaf = self._find_leaf(x_clipped)
        leaf.update(x_clipped, self.t, self.decay)

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
        """Split every dimension; redistribute historical summaries per ``split_strategy``."""
        log.debug("t=%d split leaf level=%d center=%s strategy=%s",
                  self.t, cell.level, np.round(cell.center, 3), self.split_strategy)
        mids = 0.5 * (cell.lower + cell.upper)
        entries: list[tuple[np.ndarray, GridCell]] = []
        for bits in itertools.product([0, 1], repeat=self.dim):
            bits = np.asarray(bits, dtype=int)
            lower = np.where(bits == 0, cell.lower, mids)
            upper = np.where(bits == 0, mids, cell.upper)
            entries.append((bits, GridCell(lower, upper, cell.level + 1, cell.last_update)))

        if self.split_strategy == "equal_uniform":
            self._redistribute_equal_uniform(cell, entries)
        elif self.split_strategy == "point_mass":
            self._redistribute_point_mass(cell, entries)
        else:
            self._redistribute_moment_based(cell, entries, mids)

        cell.children = [child for _, child in entries]
        cell.split_time = self.t
        # cell is now an internal node; its own stats are stale (they were
        # the pre-split parent's) and unused by any is_leaf-gated code path,
        # but leaving them set is a footgun for future code/debug logging
        # that reads a node's stats without checking is_leaf first.
        cell.s0 = 0.0
        cell.s1 = np.zeros(self.dim)
        cell.s2 = np.zeros(self.dim)
        cell.raw_count = 0

    def _redistribute_equal_uniform(self, cell: GridCell, entries: list[tuple[np.ndarray, GridCell]]) -> None:
        # v0 strategy: historical mass is split equally among children and
        # assumed locally uniform inside each child. This gives child-specific
        # moments consistent with that assumption.
        k = len(entries)
        for _, child in entries:
            child.s0 = cell.s0 / k
            mu = child.center
            var = (child.side_lengths ** 2) / 12.0
            child.s1 = child.s0 * mu
            child.s2 = child.s0 * (var + mu * mu)
            child.raw_count = cell.raw_count // k

    def _redistribute_point_mass(self, cell: GridCell, entries: list[tuple[np.ndarray, GridCell]]) -> None:
        # Degenerate strategy: treat all of the parent's accumulated mass as
        # concentrated at its tracked mean, and hand the entire history to
        # whichever single child contains that point. A useful contrast
        # against equal_uniform (spreads mass regardless of where it is) and
        # moment_based (splits mass proportionally).
        mu_clipped = np.clip(cell.mean(), cell.lower, cell.upper)
        assigned = False
        for _, child in entries:
            if not assigned and child.contains(mu_clipped):
                child.s0 = cell.s0
                child.s1 = cell.s1.copy()
                child.s2 = cell.s2.copy()
                child.raw_count = cell.raw_count
                assigned = True
            else:
                child.s0 = 0.0
                child.s1 = np.zeros(self.dim)
                child.s2 = np.zeros(self.dim)
                child.raw_count = 0

    def _redistribute_moment_based(
        self, cell: GridCell, entries: list[tuple[np.ndarray, GridCell]], mids: np.ndarray,
    ) -> None:
        # Principled strategy: model the parent's within-cell distribution as
        # an axis-independent Gaussian matching its tracked mean/variance,
        # truncated to the parent's own bounds. Each child then gets the
        # Gaussian mass its half-interval implies per axis (not an equal
        # share), and its own mean/variance are the corresponding truncated-
        # normal moments rather than an assumed-uniform placeholder.
        mu = cell.mean()
        sigma = np.sqrt(cell.variance())
        lower_frac = np.array([
            _axis_lower_fraction(mu[j], sigma[j], cell.lower[j], mids[j], cell.upper[j])
            for j in range(self.dim)
        ])
        lower_moments = [_truncated_normal_moments(mu[j], sigma[j], cell.lower[j], mids[j]) for j in range(self.dim)]
        upper_moments = [_truncated_normal_moments(mu[j], sigma[j], mids[j], cell.upper[j]) for j in range(self.dim)]

        for bits, child in entries:
            frac = 1.0
            child_mean = np.empty(self.dim)
            child_var = np.empty(self.dim)
            for j in range(self.dim):
                if bits[j] == 0:
                    frac *= lower_frac[j]
                    child_mean[j], child_var[j] = lower_moments[j]
                else:
                    frac *= (1.0 - lower_frac[j])
                    child_mean[j], child_var[j] = upper_moments[j]
            child.s0 = cell.s0 * frac
            child.s1 = child.s0 * child_mean
            child.s2 = child.s0 * (child_var + child_mean ** 2)
            child.raw_count = int(round(cell.raw_count * frac))

    def maintenance(self) -> None:
        n_leaves_before = len(self.leaves())
        self._prune_recursive(self.root)
        self._contract_recursive(self.root)
        self._assign_clusters()
        leaves = self.leaves()
        dense = self.dense_leaves()
        n_clusters = len({c.cluster_id for c in leaves if c.cluster_id is not None})
        log.info("t=%d maintenance: leaves %d->%d, dense=%d, clusters=%d",
                 self.t, n_leaves_before, len(leaves), len(dense), n_clusters)

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
                sparse = child.s0 * self._volume_scale(child) < self.sparse_threshold
                if inactive and sparse:
                    log.debug("t=%d prune idle+sparse leaf level=%d center=%s s0=%.4f",
                              self.t, child.level, np.round(child.center, 3), child.s0)
                    continue
            kept.append(child)
        node.children = kept

    def _contract_recursive(self, node: GridCell) -> None:
        """Merge a fully-split cell's children back into it when none of
        them individually still need the resolution (``max R(g_i) <
        merge_threshold``, mirroring the paper's contraction criterion).

        Bottom-up: children are checked first, so a cascade of stale splits
        can collapse in a single maintenance pass. ``merge_min_age`` guards
        against undoing a split before its children have had a chance to
        accumulate new data — right after a split every child's
        refinement score is exactly 0 by construction (its moments are set
        to match the assumed-uniform baseline that ``refinement_score``
        measures deviation from), so without this guard a cell could merge
        back on the very next maintenance call regardless of how the split
        was seeded.
        """
        if node.is_leaf:
            return
        for child in node.children:
            self._contract_recursive(child)
        if node.split_time is None or (self.t - node.split_time) < self.merge_min_age:
            return
        if not all(child.is_leaf for child in node.children):
            return
        for child in node.children:
            child.decay_to(self.t, self.decay)
        scores = [child.refinement_score(self.alpha_var, self.alpha_mean) for child in node.children]
        if max(scores, default=0.0) < self.merge_threshold:
            log.debug("t=%d contract node level=%d center=%s (max child score=%.4f < %.4f)",
                      self.t, node.level, np.round(node.center, 3), max(scores, default=0.0), self.merge_threshold)
            self._merge_children(node)

    def _merge_children(self, node: GridCell) -> None:
        children = node.children
        node.s0 = sum(c.s0 for c in children)
        node.s1 = sum((c.s1 for c in children), start=np.zeros(self.dim))
        node.s2 = sum((c.s2 for c in children), start=np.zeros(self.dim))
        node.raw_count = sum(c.raw_count for c in children)
        node.last_update = self.t
        node.children = []
        node.split_time = None

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
            if c.state(self.t, self.decay, self.dense_threshold, self.sparse_threshold,
                        self._volume_scale(c)) == "dense"
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

        # Border assignment: a transitional cell that is face-adjacent to a
        # dense cluster is attached to it (its highest-density neighbor,
        # where more than one qualifies), mirroring DBSCAN's border-point
        # rule. A transitional cell with no dense neighbor stays unassigned,
        # same as a sparse cell.
        transitional = [
            c for c in leaves
            if c.state(self.t, self.decay, self.dense_threshold, self.sparse_threshold,
                        self._volume_scale(c)) == "transitional"
        ]
        for c in transitional:
            best_neighbor = None
            for d in dense:
                if self._cells_touch_by_face(c, d) and (best_neighbor is None or d.s0 > best_neighbor.s0):
                    best_neighbor = d
            if best_neighbor is not None:
                c.cluster_id = best_neighbor.cluster_id

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
