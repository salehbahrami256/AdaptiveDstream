"""Per-cell equidistribution criteria: given a raw point sample known to lie
in a fixed axis-aligned box, produce a scalar "how far from uniform is this"
score.

This module exists to let a criterion be evaluated *offline*, from a raw
point array, independent of the streaming decayed sufficient statistics
:class:`~adaptive_dstream.cell.GridCell` maintains during a live run. That
separation is what lets :mod:`examples.run_cell_equidistribution_sweep`
compare criteria against each other on controlled synthetic shapes without
running the full split/merge tree at all.

Two criteria are provided:

``moment_refinement_score``
    The statistic already used in production (:meth:`GridCell.refinement_score`),
    recomputed here from a raw sample with equal weights (no decay) so it can
    be compared apples-to-apples against a batch of i.i.d. points. It has no
    known null distribution: its decision threshold is a tuned constant.

``weyl_equidistribution_score``
    A goodness-of-fit statistic built from Weyl sums (finite-Fourier / trigonometric
    moments) evaluated at a small set of nonzero integer frequency vectors.
    Unlike the moment score, it has an exact mean-zero, and asymptotically
    chi-square, null distribution under "the data is uniform on this box" —
    so its decision threshold is a chi-square quantile at a chosen
    significance level, not a tuned constant. See the docstring below for
    the derivation.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import chi2


def _rescale_to_unit_box(X: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    width = np.maximum(upper - lower, 1e-12)
    return (np.asarray(X, dtype=float) - lower) / width


def moment_refinement_score(X: np.ndarray, lower: np.ndarray, upper: np.ndarray,
                             alpha_var: float = 1.0, alpha_mean: float = 1.0) -> float:
    """Batch, equal-weight version of :meth:`GridCell.refinement_score`.

    Under a uniform distribution inside the box, ``Var(X_j) / (upper_j -
    lower_j)^2 = 1/12`` and ``E[X_j]`` is the box center; this scores the
    mean absolute deviation of the sample's moments from that baseline.
    """
    X = np.asarray(X, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    if len(X) == 0:
        return 0.0
    h2 = np.maximum((upper - lower) ** 2, 1e-12)
    center = 0.5 * (lower + upper)
    mu = X.mean(axis=0)
    var = X.var(axis=0)  # population (ddof=0) variance, matching cell.py's s2/s0 - mu^2
    normalized_var = var / h2
    var_term = float(np.mean(np.abs(normalized_var - (1.0 / 12.0))))
    mean_term = float(np.mean(((mu - center) ** 2) / h2))
    return alpha_var * var_term + alpha_mean * mean_term


def default_weyl_frequencies(dim: int) -> np.ndarray:
    """A frequency set whose size grows linearly (not exponentially) in
    ``dim``: one axis-aligned unit frequency per dimension (catches per-axis
    marginal non-uniformity, the same signal the moment score sees) plus, for
    every consecutive axis pair, a "sum" and "difference" frequency (catches
    correlation/diagonal structure the moment score is blind to, since it
    only ever aggregates per-axis statistics).

    No two returned frequency vectors are equal or negatives of each other
    (checked implicitly by construction), which is exactly the condition
    under which their Weyl-sum statistics are *exactly* uncorrelated under
    the uniform null (see ``weyl_equidistribution_score``), not just
    asymptotically so.
    """
    freqs = [np.eye(dim, dtype=int)[j] for j in range(dim)]
    for j in range(dim - 1):
        e_j, e_j1 = np.zeros(dim, dtype=int), np.zeros(dim, dtype=int)
        e_j[j], e_j1[j + 1] = 1, 1
        freqs.append(e_j + e_j1)  # e_j + e_{j+1}
        diff = np.zeros(dim, dtype=int)
        diff[j], diff[j + 1] = 1, -1
        freqs.append(diff)  # e_j - e_{j+1}
    return np.array(freqs)


@dataclass
class WeylTestResult:
    statistic: float
    dof: int
    p_value: float
    per_frequency_magnitude: np.ndarray  # |S_k| for each frequency, diagnostic only

    def rejects(self, alpha: float = 0.05) -> bool:
        return self.p_value < alpha


def weyl_equidistribution_score(X: np.ndarray, lower: np.ndarray, upper: np.ndarray,
                                 frequencies: np.ndarray | None = None) -> WeylTestResult:
    """Goodness-of-fit test for "X is uniform on [lower, upper]" via Weyl sums.

    Derivation. Rescale points into u in [0,1]^dim. For a nonzero integer
    frequency vector k, define the per-point character phi(u) = exp(2*pi*i*
    k.u) and its sample mean S_k = mean(phi(u_1), ..., phi(u_n)). Because the
    integer characters e^{2*pi*i*k.u} are an orthonormal basis of L^2([0,1]^d),
    for u ~ Uniform([0,1]^d):

        E[phi(u)]           = 0        exactly, for any nonzero integer k
        E[|phi(u)|^2]       = 1        (it's a unit-modulus number)
        => Var(Re phi) = Var(Im phi) = 1/2, Cov(Re phi, Im phi) = 0, exactly

    and for two distinct nonzero integer frequencies k != +-k', every
    cross second moment between (Re phi_k, Im phi_k) and (Re phi_k', Im
    phi_k') is *exactly* zero too (k -+ k' are themselves nonzero integer
    vectors, so the same orthogonality kills the cross terms). So under the
    null, by the multivariate CLT, sqrt(n)*(Re S_k, Im S_k) jointly across
    all chosen frequencies converges to independent N(0, 1/2) pairs, giving:

        T_k = 2*n*|S_k|^2  ->  chi-square(2)   asymptotically, per frequency
        T   = sum_k T_k    ->  chi-square(2*m) asymptotically, m = #frequencies

    T is the returned ``statistic``; its p-value under chi-square(2m) is a
    real significance level, not a tuned constant, which is exactly the
    calibration the mean/variance score lacks (see module docstring and
    ``moment_refinement_score``).

    Cost: O(n * m) with m = O(dim) for the default frequency set (see
    ``default_weyl_frequencies``), i.e. linear in dimension — unlike a
    literal 2^dim-orthant chi-square/entropy test, this does not blow up as
    dim grows.
    """
    X = np.asarray(X, dtype=float)
    dim = X.shape[1] if X.ndim > 1 else 1
    n = len(X)
    if frequencies is None:
        frequencies = default_weyl_frequencies(dim)
    frequencies = np.asarray(frequencies, dtype=float)
    m = len(frequencies)

    if n == 0 or m == 0:
        return WeylTestResult(statistic=0.0, dof=max(2 * m, 1), p_value=1.0,
                               per_frequency_magnitude=np.zeros(m))

    u = _rescale_to_unit_box(X, lower, upper)
    angles = 2.0 * np.pi * (u @ frequencies.T)  # (n, m)
    re = np.cos(angles).mean(axis=0)
    im = np.sin(angles).mean(axis=0)
    magnitude = np.sqrt(re ** 2 + im ** 2)
    per_freq_stat = 2.0 * n * (re ** 2 + im ** 2)
    T = float(per_freq_stat.sum())
    dof = 2 * m
    p_value = float(chi2.sf(T, df=dof))
    return WeylTestResult(statistic=T, dof=dof, p_value=p_value, per_frequency_magnitude=magnitude)


def weyl_threshold(dim: int, alpha: float = 0.05, frequencies: np.ndarray | None = None) -> float:
    """The chi-square(2m) critical value the Weyl statistic should be
    compared against for a target false-positive rate ``alpha`` — the
    principled analogue of ``AdaptiveDStream.split_threshold`` for this
    criterion. ``m`` follows ``frequencies`` if given, else
    ``len(default_weyl_frequencies(dim))``.
    """
    m = len(frequencies) if frequencies is not None else len(default_weyl_frequencies(dim))
    return float(chi2.ppf(1.0 - alpha, df=2 * m))


def default_random_directions(dim: int, n_directions: int | None = None, seed: int | None = None) -> np.ndarray:
    """``n_directions`` fixed random unit vectors in R^dim (Gaussian, then
    normalized — the standard way to draw a direction uniform on the unit
    sphere). Fixed by a seed derived from ``dim`` alone (unless overridden)
    so the criterion is deterministic and reproducible across calls, the
    same role ``default_weyl_frequencies`` plays for the Weyl test.

    Defaults ``n_directions`` to ``len(default_weyl_frequencies(dim))`` so
    the two criteria are compared under an equal O(dim) "frequency/direction
    budget" rather than one getting more tests than the other.
    """
    if n_directions is None:
        n_directions = len(default_weyl_frequencies(dim))
    rng = np.random.default_rng(seed if seed is not None else 20240000 + dim)
    v = rng.normal(size=(n_directions, dim))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return v


@dataclass
class RandomProjectionTestResult:
    statistic: float           # -log(p_bound); monotonic in how far any direction deviated
    p_value: float             # conservative (Bonferroni/Hoeffding) upper bound on the true p-value
    n_directions: int
    per_direction_mean_gap: np.ndarray   # |sample mean - 1/2 * sum(v)| per direction, diagnostic only
    per_direction_var_gap: np.ndarray    # |sample var  - 1/12|            per direction, diagnostic only

    def rejects(self, alpha: float = 0.05) -> bool:
        return self.p_value < alpha


def random_projection_score(X: np.ndarray, lower: np.ndarray, upper: np.ndarray,
                             directions: np.ndarray | None = None) -> RandomProjectionTestResult:
    """Goodness-of-fit test for "X is uniform on [lower, upper]" via random
    linear projections, with an exact finite-sample false-positive bound
    (Hoeffding's inequality + a union bound across directions and the two
    sub-tests run on each) rather than an asymptotic approximation.

    Derivation. Rescale points into u in [0,1]^dim. For a fixed unit vector
    v (||v||_2 = 1), let Y = v.u. Under u ~ Uniform([0,1]^dim):

        E[Y]      = 0.5 * sum(v)                         (exact)
        Var(Y)    = sum(v_j^2 * Var(u_j)) = ||v||^2 / 12 = 1/12   (exact,
                     independent of v and dim, since ||v||=1)
        Y in [a(v), b(v)],  a(v) = sum(min(v_j,0)), b(v) = sum(max(v_j,0)),
                     range R(v) = b(v) - a(v) = sum(|v_j|)

    Hoeffding's inequality (1963): for i.i.d. Y_1..Y_n in [a,b] with mean mu,
    for any t > 0,  P(|Ybar_n - mu| >= t) <= 2*exp(-2*n*t^2 / (b-a)^2).

    This is applied twice per direction, both times to a mean of bounded
    i.i.d. variables (so Hoeffding applies directly, with no need for the
    projected distribution's exact shape):

      - MEAN sub-test: Y_i itself, comparing Ybar_n to the exact null mean
        0.5*sum(v).
      - VARIANCE sub-test: Z_i = (Y_i - 0.5*sum(v))^2 in [0, R(v)^2] (using
        the *known* null mean, not the sample mean, so Z_i stays an i.i.d.
        bounded sequence), comparing Zbar_n to the exact null variance 1/12.

    With m directions, that is 2m elementary tests; each is run at level
    alpha/(2m) and the union bound gives an exact, finite-sample (not just
    asymptotic) guarantee: P(any elementary test rejects | H0) <= alpha, at
    *every* n, not only as n -> infinity. The price for that non-asymptotic
    guarantee is that Hoeffding's bound is loose (it uses only boundedness,
    not the true variance), so at equal alpha this test is expected to have
    materially less power than the asymptotically-exact Weyl/chi-square
    test at any fixed, finite n -- a real, measurable trade-off (see
    examples/run_cell_equidistribution_sweep.py), not just a footnote.

    Also unlike the Weyl test, a *linear* projection's mean is blind to any
    alternative that is symmetric about the box center along every tested
    direction (e.g. diagonal_correlated concentrates around the center
    without shifting it) -- the variance sub-test above exists specifically
    to recover sensitivity to that kind of spread/concentration change,
    since the mean sub-test alone cannot see it.

    Cost: O(n * m), m = O(dim) for the default direction set -- linear in
    dimension, like the Weyl test.
    """
    X = np.asarray(X, dtype=float)
    dim = X.shape[1] if X.ndim > 1 else 1
    n = len(X)
    if directions is None:
        directions = default_random_directions(dim)
    directions = np.asarray(directions, dtype=float)
    m = len(directions)

    if n == 0 or m == 0:
        return RandomProjectionTestResult(statistic=0.0, p_value=1.0, n_directions=m,
                                           per_direction_mean_gap=np.zeros(m),
                                           per_direction_var_gap=np.zeros(m))

    u = _rescale_to_unit_box(X, lower, upper)
    Y = u @ directions.T  # (n, m)
    mu0 = 0.5 * directions.sum(axis=1)          # (m,)
    R = np.abs(directions).sum(axis=1)          # (m,), exact range of Y per direction
    R = np.maximum(R, 1e-12)

    mean_gap = np.abs(Y.mean(axis=0) - mu0)                       # (m,)
    Z = (Y - mu0) ** 2                                            # (n, m), in [0, R^2]
    var_gap = np.abs(Z.mean(axis=0) - (1.0 / 12.0))               # (m,)

    # Hoeffding tail bound for each of the 2m elementary tests, evaluated at
    # the *observed* gap -- a valid (conservative) per-test p-value bound.
    p_mean = np.minimum(1.0, 2.0 * np.exp(-2.0 * n * mean_gap ** 2 / R ** 2))
    p_var = np.minimum(1.0, 2.0 * np.exp(-2.0 * n * var_gap ** 2 / R ** 4))

    p_min = float(min(p_mean.min(), p_var.min()))
    p_bound = min(1.0, 2.0 * m * p_min)  # Bonferroni over the 2m elementary tests
    statistic = -np.log(max(p_bound, 1e-300))
    return RandomProjectionTestResult(statistic=statistic, p_value=p_bound, n_directions=m,
                                       per_direction_mean_gap=mean_gap, per_direction_var_gap=var_gap)
