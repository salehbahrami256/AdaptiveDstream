from __future__ import annotations
import numpy as np


def make_drifting_stream(n_samples: int = 3000, random_state: int = 42):
    """Two 2D Gaussian clusters whose locations/scales change in three phases."""
    rng = np.random.default_rng(random_state)
    X = np.zeros((n_samples, 2), dtype=float)
    y = np.zeros(n_samples, dtype=int)
    phase = np.zeros(n_samples, dtype=int)
    cuts = [0, n_samples // 3, 2 * n_samples // 3, n_samples]
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
