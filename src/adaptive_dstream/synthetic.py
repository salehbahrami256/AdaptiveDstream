from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


def _orbit_centers(dim: int, n_phases: int, radius: float = 3.0):
    """Place an antipodal pair of per-phase centers orbiting the origin.

    The orbit lives in the first two coordinates (first one only if
    ``dim == 1``); any remaining coordinates are held at zero. This keeps
    the "two clusters, always separated, drifting" shape meaningful in any
    dimension >= 1 while staying visualizable via its leading axes.
    """
    angles = np.linspace(0, 2 * np.pi, n_phases, endpoint=False)
    centers_a, centers_b = [], []
    for a in angles:
        ca, cb = np.zeros(dim), np.zeros(dim)
        ca[0] = radius * np.cos(a)
        cb[0] = -radius * np.cos(a)
        if dim > 1:
            ca[1] = radius * np.sin(a)
            cb[1] = -radius * np.sin(a)
        centers_a.append(ca)
        centers_b.append(cb)
    return centers_a, centers_b


def make_drifting_stream(n_samples: int = 3000, random_state: int = 42, dim: int = 2):
    """Two Gaussian clusters whose locations/scales change in three phases.

    ``dim`` defaults to 2 (the hand-tuned configuration used by the
    quickstart visualization, kept bit-for-bit reproducible). Other
    dimensions fall back to an isotropic, dimension-generic layout via
    :func:`_orbit_centers`.
    """
    rng = np.random.default_rng(random_state)
    X = np.zeros((n_samples, dim), dtype=float)
    y = np.zeros(n_samples, dtype=int)
    phase = np.zeros(n_samples, dtype=int)
    cuts = [0, n_samples // 3, 2 * n_samples // 3, n_samples]

    if dim == 2:
        configs = [
            ([(-2.5, -1.0), (2.5, 1.0)], [(0.45, 0.55), (0.55, 0.45)]),
            ([(-1.2, 1.8), (1.4, -1.5)], [(0.55, 0.40), (0.45, 0.65)]),
            ([(-2.0, 2.2), (2.2, 2.0)], [(0.35, 0.80), (0.80, 0.35)]),
        ]
        for p in range(3):
            means, scales = configs[p]
            for i in range(cuts[p], cuts[p + 1]):
                label = int(rng.random() > 0.5)
                X[i] = rng.normal(np.array(means[label]), np.array(scales[label]))
                y[i] = label
                phase[i] = p
        return X, y, phase

    centers0, centers1 = _orbit_centers(dim, n_phases=3, radius=2.5)
    for p in range(3):
        for i in range(cuts[p], cuts[p + 1]):
            label = int(rng.random() > 0.5)
            mu = centers0[p] if label == 0 else centers1[p]
            X[i] = rng.normal(mu, 0.5)
            y[i] = label
            phase[i] = p
    return X, y, phase


def make_varying_density_stream(
    n_samples: int = 3000,
    random_state: int = 42,
    dense_std: float = 0.18,
    sparse_std: float = 1.6,
    dense_weight: float = 0.5,
    n_phases: int = 3,
    dim: int = 2,
):
    """A stream with one tight/dense cluster and one broad/sparse cluster
    present *simultaneously* at every time step, drifting over ``n_phases``
    phases.

    This is the key adversarial case for any grid-based streaming method
    with a single global cell size: a cell small enough to resolve the tight
    cluster is wasteful (mostly empty) over the broad cluster's footprint,
    while a cell large enough to keep the broad cluster's density above its
    dense-threshold will merge the tight cluster into a handful of cells and
    lose its internal structure, or worse, bridge the two clusters together
    if they ever drift close.

    Label 0 is always the dense/tight cluster, label 1 the sparse/broad one.
    ``dim`` (default 2, matching the reproducible ``seed=7`` results in the
    README) generalizes to any dimension >= 1; the two clusters' centers
    orbit the origin in the leading one or two axes via :func:`_orbit_centers`
    for ``dim != 2`` or ``n_phases != 3`` (the dim=2/n_phases=3 case keeps
    its original hand-picked centers for exact backward compatibility).
    """
    rng = np.random.default_rng(random_state)
    X = np.zeros((n_samples, dim), dtype=float)
    y = np.zeros(n_samples, dtype=int)
    phase = np.zeros(n_samples, dtype=int)
    cuts = np.linspace(0, n_samples, n_phases + 1).astype(int)

    # Both cluster centers drift along independent smooth paths so that at
    # some point during the run their footprints partially overlap in space
    # even though their local densities never do.
    if dim == 2 and n_phases == 3:
        dense_centers = [np.array(c) for c in [(-2.2, -2.0), (0.0, -0.5), (2.0, 1.5)]]
        sparse_centers = [np.array(c) for c in [(2.0, 2.0), (0.5, 1.0), (-1.5, -1.5)]]
    else:
        dense_centers, sparse_centers = _orbit_centers(dim, n_phases, radius=3.0)

    for p in range(n_phases):
        lo, hi = cuts[p], cuts[p + 1]
        if hi <= lo:
            continue
        dense_mu = np.array(dense_centers[p % len(dense_centers)])
        sparse_mu = np.array(sparse_centers[p % len(sparse_centers)])
        for i in range(lo, hi):
            label = int(rng.random() > dense_weight)
            if label == 0:
                X[i] = rng.normal(dense_mu, dense_std)
            else:
                X[i] = rng.normal(sparse_mu, sparse_std)
            y[i] = label
            phase[i] = p
    return X, y, phase


def make_moons_stream(n_samples: int = 3000, random_state: int = 42, noise: float = 0.08, n_phases: int = 3,
                       dim: int = 2):
    """Two interleaving, non-convex 'moon' clusters that rotate and drift
    over ``n_phases`` phases. Exercises arbitrary-shape (non-Gaussian, non-
    convex) clusters rather than only isotropic blobs.

    The moon shape itself is inherently 2D (``sklearn.datasets.make_moons``
    plus a 2D rotation), so ``dim`` (default 2, exactly reproducing the
    original stream) requires ``dim >= 2``; any extra dimensions beyond the
    first two are filled with small isotropic Gaussian noise so the moons
    are embedded in, rather than reshaped by, the higher-dimensional space.
    """
    from sklearn.datasets import make_moons

    if dim < 2:
        raise ValueError("make_moons_stream requires dim >= 2 (the moon shape is inherently 2D).")

    rng = np.random.default_rng(random_state)
    X = np.zeros((n_samples, dim), dtype=float)
    y = np.zeros(n_samples, dtype=int)
    phase = np.zeros(n_samples, dtype=int)
    cuts = np.linspace(0, n_samples, n_phases + 1).astype(int)

    for p in range(n_phases):
        lo, hi = cuts[p], cuts[p + 1]
        n = hi - lo
        if n <= 0:
            continue
        seed = int(rng.integers(0, 2**32 - 1))
        Xp, yp = make_moons(n_samples=n, noise=noise, random_state=seed)
        Xp *= 2.5
        angle = p * (np.pi / n_phases)
        rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        Xp = Xp @ rot.T
        shift = np.array([1.5 * np.cos(2 * np.pi * p / n_phases), 1.5 * np.sin(2 * np.pi * p / n_phases)])
        Xp = Xp + shift
        X[lo:hi, :2] = Xp
        if dim > 2:
            X[lo:hi, 2:] = rng.normal(0.0, noise, size=(n, dim - 2))
        y[lo:hi] = yp
        phase[lo:hi] = p
    return X, y, phase


GENERATORS = {
    "two_gaussian": make_drifting_stream,
    "varying_density": make_varying_density_stream,
    "moons": make_moons_stream,
}


def generate_stream(kind: str, n_samples: int = 3000, random_state: int = 42, **kwargs):
    """Dispatch to one of the named generators in ``GENERATORS``."""
    if kind not in GENERATORS:
        raise ValueError(f"Unknown stream kind {kind!r}; choose from {sorted(GENERATORS)}")
    return GENERATORS[kind](n_samples=n_samples, random_state=random_state, **kwargs)


@dataclass
class StreamManifest:
    """Everything needed to reproduce a saved synthetic stream exactly."""

    generator: str
    seed: int
    params: dict = field(default_factory=dict)
    n_samples: int = 0
    dim: int = 0
    created_at: str = ""

    def to_dict(self) -> dict:
        return {
            "generator": self.generator,
            "seed": self.seed,
            "params": self.params,
            "n_samples": self.n_samples,
            "dim": self.dim,
            "created_at": self.created_at,
        }


def save_stream(out_dir: str | Path, name: str, X: np.ndarray, y: np.ndarray, phase: np.ndarray,
                 generator: str, seed: int, params: dict | None = None) -> Path:
    """Persist a generated stream (arrays + a JSON manifest carrying the
    generator name, seed, and parameters) so the exact same data can either
    be reloaded byte-for-byte or regenerated from scratch via
    ``generate_stream(generator, random_state=seed, **params)``.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_path = out_dir / f"{name}.npz"
    manifest_path = out_dir / f"{name}.json"

    np.savez_compressed(data_path, X=X, y=y, phase=phase)
    manifest = StreamManifest(
        generator=generator,
        seed=seed,
        params=params or {},
        n_samples=int(len(X)),
        dim=int(X.shape[1]) if X.ndim > 1 else 1,
        created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )
    manifest_path.write_text(json.dumps(manifest.to_dict(), indent=2))
    return data_path


def load_stream(out_dir: str | Path, name: str):
    """Load arrays saved by :func:`save_stream` plus their manifest dict."""
    out_dir = Path(out_dir)
    data = np.load(out_dir / f"{name}.npz")
    manifest = json.loads((out_dir / f"{name}.json").read_text())
    return data["X"], data["y"], data["phase"], manifest


def regenerate_from_manifest(manifest: dict):
    """Recreate a stream from a manifest dict, verifying reproducibility."""
    return generate_stream(manifest["generator"], n_samples=manifest["n_samples"],
                            random_state=manifest["seed"], **manifest.get("params", {}))
