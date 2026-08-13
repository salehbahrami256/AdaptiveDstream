from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class GridCell:
    lower: np.ndarray
    upper: np.ndarray
    level: int
    last_update: int = 0
    s0: float = 0.0
    s1: np.ndarray | None = None
    s2: np.ndarray | None = None
    raw_count: int = 0
    children: list["GridCell"] = field(default_factory=list)
    cluster_id: Optional[int] = None

    def __post_init__(self) -> None:
        self.lower = np.asarray(self.lower, dtype=float)
        self.upper = np.asarray(self.upper, dtype=float)
        d = len(self.lower)
        if self.s1 is None:
            self.s1 = np.zeros(d, dtype=float)
        if self.s2 is None:
            self.s2 = np.zeros(d, dtype=float)

    @property
    def dim(self) -> int:
        return len(self.lower)

    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0

    @property
    def center(self) -> np.ndarray:
        return 0.5 * (self.lower + self.upper)

    @property
    def side_lengths(self) -> np.ndarray:
        return self.upper - self.lower

    def contains(self, x: np.ndarray) -> bool:
        return bool(np.all(x >= self.lower) and np.all(x <= self.upper))

    def decay_to(self, t: int, decay: float) -> None:
        if t < self.last_update:
            raise ValueError("Time must be non-decreasing.")
        dt = t - self.last_update
        if dt > 0:
            factor = decay ** dt
            self.s0 *= factor
            self.s1 *= factor
            self.s2 *= factor
            self.last_update = t

    def update(self, x: np.ndarray, t: int, decay: float) -> None:
        self.decay_to(t, decay)
        self.s0 += 1.0
        self.s1 += x
        self.s2 += x * x
        self.raw_count += 1

    def mean(self) -> np.ndarray:
        if self.s0 <= 1e-12:
            return self.center.copy()
        return self.s1 / self.s0

    def variance(self) -> np.ndarray:
        if self.s0 <= 1e-12:
            return np.zeros(self.dim)
        mu = self.mean()
        var = self.s2 / self.s0 - mu * mu
        return np.maximum(var, 0.0)

    def refinement_score(self, alpha_var: float = 1.0, alpha_mean: float = 1.0) -> float:
        h2 = np.maximum(self.side_lengths ** 2, 1e-12)
        # Under a uniform distribution inside an axis-aligned cell,
        # Var(X_j) / h_j^2 = 1/12 and E[X_j] is the cell center.
        # We therefore refine when the maintained moments deviate from
        # this coarse within-cell approximation.
        normalized_var = self.variance() / h2
        var_term = float(np.mean(np.abs(normalized_var - (1.0 / 12.0))))
        mean_term = float(np.mean(((self.mean() - self.center) ** 2) / h2))
        return alpha_var * var_term + alpha_mean * mean_term

    def state(self, t: int, decay: float, dense_threshold: float, sparse_threshold: float) -> str:
        self.decay_to(t, decay)
        if self.s0 >= dense_threshold:
            return "dense"
        if self.s0 < sparse_threshold:
            return "sparse"
        return "transitional"
