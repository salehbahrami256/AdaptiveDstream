"""Single-cell equidistribution test shapes.

Unlike :mod:`adaptive_dstream.synthetic` (whole streams with labels/phases
for the end-to-end clustering evaluation), everything here answers a
narrower question: "here is one box and a batch of n points inside it — is
the local distribution equidistributed or not, and in what way not?" Every
generator returns a plain ``(n, dim)`` array of points in the unit box
``[0, 1]^dim``; the null (``sample_uniform``) is the H0 every other shape is
tested against.

The shapes are chosen to be adversarial in specific, named ways:

- ``sample_uniform``: H0 baseline.
- ``sample_core_border_imbalance``: a uniform core plus two thin border
  slabs at *opposite* faces of the box with different densities from each
  other (and from the core) — the "cluster center looks fine, its two edges
  don't agree with each other" case from the paper discussion. ``axis="diag"``
  places the border along the main diagonal instead of a coordinate axis via
  a shared additive shift on every coordinate. Empirically this does *not*
  end up axis-blind (the shared shift still perturbs every per-axis
  marginal, so both criteria detect it at AUC ~1 across dimensions in
  practice) — ``sample_diagonal_correlated`` below is the shape that
  actually isolates the axis-blind case cleanly; this one is kept as an
  easier sanity-check variant of the border-imbalance idea.
- ``sample_diagonal_correlated``: points concentrated in a thin band around
  the main diagonal. Every 1-D marginal can look close to uniform while the
  joint distribution is highly concentrated — the classic blind spot of any
  test that only ever looks at one axis at a time.
- ``sample_symmetric_bimodal``: two symmetric blobs straddling the box
  center along one axis. Mean equals the center and variance can be tuned
  close to the uniform value 1/12 exactly while the shape is obviously not
  uniform — the "same moments, different shape" case a variance-only
  criterion cannot see by construction.
- ``sample_corner_spike``: a point-mass concentration in one corner plus an
  otherwise-uniform background — the heavy-hitter case Count-Min-style
  sketches are aimed at.
"""
from __future__ import annotations

import numpy as np

SHAPES = (
    "uniform",
    "core_border_imbalance",
    "core_border_imbalance_diag",
    "diagonal_correlated",
    "symmetric_bimodal",
    "corner_spike",
)


def sample_uniform(n: int, dim: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(0.0, 1.0, size=(n, dim))


def sample_core_border_imbalance(n: int, dim: int, seed: int, border_width: float = 0.15,
                                  low_weight: float = 4.0, high_weight: float = 1.0,
                                  axis: int | str = 0) -> np.ndarray:
    """Uniform core, imbalanced opposite borders.

    Along ``axis`` (an int coordinate, or ``"diag"`` for the normalized main
    diagonal direction), the box is split into three regions: a "low" border
    slab of width ``border_width``, a uniform core, and a "high" border slab
    of width ``border_width``. Each region is sampled with density
    proportional to its own weight (``low_weight``/``1.0``/``high_weight``)
    times its length, so the core alone is exactly uniform while the two
    borders disagree with each other by a factor ``low_weight/high_weight``.
    """
    rng = np.random.default_rng(seed)
    bw = border_width
    lengths = np.array([bw, 1.0 - 2 * bw, bw])
    weights = np.array([low_weight, 1.0, high_weight]) * lengths
    probs = weights / weights.sum()
    region = rng.choice(3, size=n, p=probs)

    t = np.empty(n)
    t[region == 0] = rng.uniform(0.0, bw, size=(region == 0).sum())
    t[region == 1] = rng.uniform(bw, 1.0 - bw, size=(region == 1).sum())
    t[region == 2] = rng.uniform(1.0 - bw, 1.0, size=(region == 2).sum())

    X = rng.uniform(0.0, 1.0, size=(n, dim))
    if axis == "diag":
        # Move each point along the (1,...,1)/sqrt(dim) direction so its
        # diagonal coordinate (mean of coordinates) equals t, leaving the
        # perpendicular component random/uniform-ish. Simplest exact
        # construction: start from a uniform point, then shift every
        # coordinate by the same amount so the mean becomes t.
        shift = t - X.mean(axis=1)
        X = np.clip(X + shift[:, None], 0.0, 1.0)
    else:
        X[:, axis] = t
    return X


def sample_diagonal_correlated(n: int, dim: int, seed: int, band_width: float = 0.08) -> np.ndarray:
    """Points concentrated in a thin band around the main diagonal
    ``x_1 = x_2 = ... = x_dim``. Marginals stay close to uniform; the joint
    distribution is highly concentrated on a 1-D subset of the box.
    """
    rng = np.random.default_rng(seed)
    t = rng.uniform(0.0, 1.0, size=(n, 1))
    noise = rng.uniform(-band_width / 2, band_width / 2, size=(n, dim))
    return np.clip(t + noise, 0.0, 1.0)


def sample_symmetric_bimodal(n: int, dim: int, seed: int, separation: float = 0.25,
                              blob_std: float = 0.08, axis: int = 0) -> np.ndarray:
    """Two Gaussian blobs symmetric about the box center along ``axis``;
    every other axis is uniform over the full box. Mean is exactly the box
    center by symmetry; ``separation``/``blob_std`` can be tuned so the
    axis-``axis`` variance sits near uniform's 1/12 while the shape is
    obviously bimodal, not uniform.
    """
    rng = np.random.default_rng(seed)
    X = rng.uniform(0.0, 1.0, size=(n, dim))
    side = rng.integers(0, 2, size=n)
    center = 0.5 + np.where(side == 0, -separation, separation)
    X[:, axis] = np.clip(rng.normal(center, blob_std), 0.0, 1.0)
    return X


def sample_corner_spike(n: int, dim: int, seed: int, spike_frac: float = 0.3,
                         spike_width: float = 0.08) -> np.ndarray:
    """A point-mass concentration in the ``[0, spike_width]^dim`` corner,
    otherwise uniform background — the heavy-hitter / point-mass case.
    """
    rng = np.random.default_rng(seed)
    X = rng.uniform(0.0, 1.0, size=(n, dim))
    is_spike = rng.random(n) < spike_frac
    n_spike = int(is_spike.sum())
    X[is_spike] = rng.uniform(0.0, spike_width, size=(n_spike, dim))
    return X


GENERATORS = {
    "uniform": sample_uniform,
    "core_border_imbalance": sample_core_border_imbalance,
    "core_border_imbalance_diag": lambda n, dim, seed: sample_core_border_imbalance(n, dim, seed, axis="diag"),
    "diagonal_correlated": sample_diagonal_correlated,
    "symmetric_bimodal": sample_symmetric_bimodal,
    "corner_spike": sample_corner_spike,
}


def sample_shape(kind: str, n: int, dim: int, seed: int) -> np.ndarray:
    if kind not in GENERATORS:
        raise ValueError(f"Unknown shape {kind!r}; choose from {sorted(GENERATORS)}")
    return GENERATORS[kind](n, dim, seed)
