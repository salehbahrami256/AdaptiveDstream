"""Adaptive-radius micro-cluster stream clustering (``AdaptiveRadiusMC``).

The successor to the grid-based :class:`~adaptive_dstream.model.AdaptiveDStream`
on this branch. The grid method's central idea -- resolution that adapts to
the local density -- is kept, but the mechanism changes: instead of recursively
halving cells (which fragments a region's mass across ``2**d`` children and
hits an exponential wall in ``d``), the model keeps a bounded set of
micro-clusters whose *absorption radius is local*: each micro-cluster's radius
is a multiple of its own (shrunk-toward-a-prior) standard deviation.

Consequences, relative to a single global radius (DBSTREAM's
``clustering_threshold``):

- a tight cluster's micro-clusters shrink their radius to the data scale, so
  later points just outside it spawn new micro-clusters at a finer scale (the
  analogue of a split, with no mass fragmentation);
- a broad cluster's micro-clusters grow their radius to the data scale, so it
  is covered by a few large micro-clusters instead of many weak ones;
- memory is bounded by ``max_micro`` by construction and cost is linear in
  ``d`` (no ``2**d`` children).

Same prequential interface as every other model in the harness:
``partial_fit(x, t)``, ``predict_point_cluster(x)``, ``summary()``. Clustering
runs at maintenance checkpoints (like the grid models), so the first
``maintenance_interval`` predictions are unassigned -- a cold start shared with
the grid baselines.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial.distance import cdist

from .logging_utils import get_logger

log = get_logger("microcluster")


@dataclass
class AdaptiveRadiusMC:
    init_radius: float = 0.5
    min_radius: float = 0.02
    max_radius: float = 3.0
    radius_scale: float = 2.5          # radius = radius_scale * sigma_eff
    prior_weight: float = 3.0          # pseudo-count pulling sigma toward init_radius/radius_scale
    decay: float = 0.99
    dense_weight: float = 3.0          # micro-cluster weight needed to seed/propagate a cluster
    sparse_weight: float = 0.1         # weight below which a micro-cluster is pruned
    merge_factor: float = 0.5          # merge pairs closer than merge_factor * min(radius)
    connect_factor: float = 1.0        # connect dense pairs closer than connect_factor * (r_i + r_j)
    assign_tolerance: float = 2.0      # predict uses nearest mc within this many radii
    max_micro: int = 500
    maintenance_interval: int = 50

    t: int = field(default=0, init=False)
    s0: np.ndarray = field(default=None, init=False, repr=False)
    s1: np.ndarray = field(default=None, init=False, repr=False)
    s2: np.ndarray = field(default=None, init=False, repr=False)
    labels: np.ndarray = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if not (0 < self.min_radius <= self.init_radius <= self.max_radius):
            raise ValueError("need 0 < min_radius <= init_radius <= max_radius.")
        if not (0.0 < self.decay <= 1.0):
            raise ValueError("decay must be in (0, 1].")
        if self.max_micro < 1:
            raise ValueError("max_micro must be >= 1.")
        log.info("AdaptiveRadiusMC: init_radius=%.3g radius range=[%.3g, %.3g] scale=%.2f decay=%.3f "
                 "dense_weight=%.2f max_micro=%d", self.init_radius, self.min_radius, self.max_radius,
                 self.radius_scale, self.decay, self.dense_weight, self.max_micro)

    # ------------------------------------------------------------------ state
    @property
    def n_micro(self) -> int:
        return 0 if self.s0 is None else len(self.s0)

    def _centers(self) -> np.ndarray:
        return self.s1 / self.s0[:, None]

    def _radii(self) -> np.ndarray:
        d = self.s1.shape[1]
        c = self._centers()
        var = np.maximum(self.s2 / self.s0 - np.einsum("ij,ij->i", c, c), 0.0) / d
        sigma0_sq = (self.init_radius / self.radius_scale) ** 2
        sigma_sq = (self.s0 * var + self.prior_weight * sigma0_sq) / (self.s0 + self.prior_weight)
        return np.clip(self.radius_scale * np.sqrt(sigma_sq), self.min_radius, self.max_radius)

    # ----------------------------------------------------------------- update
    def partial_fit(self, x: np.ndarray, t: int | None = None) -> "AdaptiveRadiusMC":
        x = np.asarray(x, dtype=float)
        t_new = self.t + 1 if t is None else t
        dt = max(t_new - self.t, 0)
        self.t = t_new
        if self.s0 is None:
            self.s0 = np.empty(0)
            self.s1 = np.empty((0, x.shape[0]))
            self.s2 = np.empty(0)
            self.labels = np.empty(0, dtype=int)
        elif dt and self.decay < 1.0:
            f = self.decay ** dt
            self.s0 = self.s0 * f
            self.s1 = self.s1 * f
            self.s2 = self.s2 * f

        absorbed = False
        if self.n_micro:
            dist = np.linalg.norm(self._centers() - x, axis=1)
            inside = np.where(dist <= self._radii())[0]
            if len(inside):
                i = inside[np.argmin(dist[inside])]
                self.s0[i] += 1.0
                self.s1[i] += x
                self.s2[i] += float(x @ x)
                absorbed = True
        if not absorbed:
            self.s0 = np.append(self.s0, 1.0)
            self.s1 = np.vstack([self.s1, x])
            self.s2 = np.append(self.s2, float(x @ x))
            self.labels = np.append(self.labels, -1)

        if self.t % self.maintenance_interval == 0:
            self.maintenance()
        return self

    def fit_stream(self, X: np.ndarray) -> "AdaptiveRadiusMC":
        for x in X:
            self.partial_fit(x)
        return self

    # ------------------------------------------------------------ maintenance
    def maintenance(self) -> None:
        if not self.n_micro:
            return
        before = self.n_micro
        self._prune()
        merged = self._merge_overlapping()
        capped = self._enforce_cap()
        self._assign_clusters()
        log.info("maintenance t=%d: micro %d -> %d (pruned/merged/capped, %d merged, %d capped), "
                 "dense=%d clusters=%d", self.t, before, self.n_micro, merged, capped,
                 int((self.s0 >= self.dense_weight).sum()) if self.n_micro else 0,
                 len({int(v) for v in self.labels if v >= 0}))

    def _keep(self, mask: np.ndarray) -> None:
        self.s0, self.s1, self.s2, self.labels = (
            self.s0[mask], self.s1[mask], self.s2[mask], self.labels[mask])

    def _prune(self) -> None:
        keep = self.s0 >= self.sparse_weight
        if not keep.all():
            log.debug("prune: dropping %d micro-clusters below weight %.3g", int((~keep).sum()), self.sparse_weight)
            self._keep(keep)

    def _merge_overlapping(self) -> int:
        m = self.n_micro
        if m < 2:
            return 0
        c, r = self._centers(), self._radii()
        dist = cdist(c, c)
        limit = self.merge_factor * np.minimum(r[:, None], r[None, :])
        ii, jj = np.where(np.triu(dist <= limit, k=1))
        if not len(ii):
            return 0
        order = np.argsort(dist[ii, jj])
        dead = np.zeros(m, dtype=bool)
        n_merged = 0
        for k in order:
            i, j = ii[k], jj[k]
            if dead[i] or dead[j]:
                continue
            # keep the heavier one, fold the other into it (stats are additive)
            a, b = (i, j) if self.s0[i] >= self.s0[j] else (j, i)
            self.s0[a] += self.s0[b]
            self.s1[a] += self.s1[b]
            self.s2[a] += self.s2[b]
            dead[b] = True
            n_merged += 1
        self._keep(~dead)
        return n_merged

    def _enforce_cap(self) -> int:
        excess = self.n_micro - self.max_micro
        if excess <= 0:
            return 0
        drop = np.argsort(self.s0)[:excess]
        keep = np.ones(self.n_micro, dtype=bool)
        keep[drop] = False
        self._keep(keep)
        return excess

    # -------------------------------------------------------------- clustering
    def _assign_clusters(self) -> None:
        m = self.n_micro
        self.labels = np.full(m, -1, dtype=int)
        dense = np.where(self.s0 >= self.dense_weight)[0]
        if not len(dense):
            return
        c, r = self._centers(), self._radii()
        cd, rd = c[dense], r[dense]
        dist = cdist(cd, cd)
        adj = dist <= self.connect_factor * (rd[:, None] + rd[None, :])
        comp = -np.ones(len(dense), dtype=int)
        n_comp = 0
        for s in range(len(dense)):
            if comp[s] >= 0:
                continue
            stack = [s]
            comp[s] = n_comp
            while stack:
                u = stack.pop()
                for v in np.where(adj[u] & (comp < 0))[0]:
                    comp[v] = n_comp
                    stack.append(v)
            n_comp += 1
        self.labels[dense] = comp
        # border rule: a light micro-cluster joins the nearest reachable dense one
        light = np.where(self.s0 < self.dense_weight)[0]
        for i in light:
            dd = np.linalg.norm(cd - c[i], axis=1)
            reach = self.connect_factor * (rd + r[i])
            ok = np.where(dd <= reach)[0]
            if len(ok):
                self.labels[i] = comp[ok[np.argmin(dd[ok])]]

    # -------------------------------------------------------------- prediction
    def predict_point_cluster(self, x: np.ndarray) -> int | None:
        if not self.n_micro:
            return None
        x = np.asarray(x, dtype=float)
        norm = np.linalg.norm(self._centers() - x, axis=1) / self._radii()
        i = int(np.argmin(norm))
        if norm[i] > self.assign_tolerance or self.labels[i] < 0:
            return None
        return int(self.labels[i])

    def summary(self) -> dict:
        dense = int((self.s0 >= self.dense_weight).sum()) if self.n_micro else 0
        # read-only: summary() must not re-cluster, or evaluation snapshots would
        # change predictions between checkpoints.
        n_clusters = len({int(v) for v in self.labels if v >= 0}) if self.n_micro else 0
        return {"n_leaves": self.n_micro, "n_dense_leaves": dense, "n_clusters": n_clusters}
