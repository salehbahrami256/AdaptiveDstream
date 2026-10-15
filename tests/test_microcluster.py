import numpy as np
import pytest

from adaptive_dstream import AdaptiveRadiusMC


def _blobs(n, centers, std, seed=0):
    rng = np.random.default_rng(seed)
    k = rng.integers(0, len(centers), n)
    return np.asarray(centers)[k] + rng.normal(0, std, (n, len(centers[0]))), k


def test_rejects_invalid_parameters():
    with pytest.raises(ValueError):
        AdaptiveRadiusMC(init_radius=0.001, min_radius=0.1)
    with pytest.raises(ValueError):
        AdaptiveRadiusMC(decay=1.5)
    with pytest.raises(ValueError):
        AdaptiveRadiusMC(max_micro=0)


def test_predict_before_any_data_is_none():
    assert AdaptiveRadiusMC().predict_point_cluster(np.zeros(2)) is None


def test_nearby_points_are_absorbed_distant_point_spawns_new_micro():
    m = AdaptiveRadiusMC(init_radius=0.5, maintenance_interval=10_000)
    m.partial_fit(np.array([0.0, 0.0]))
    m.partial_fit(np.array([0.05, 0.0]))
    assert m.n_micro == 1
    m.partial_fit(np.array([10.0, 10.0]))
    assert m.n_micro == 2


def test_sufficient_statistics_conserve_weight_without_decay():
    X, _ = _blobs(300, [[0, 0], [5, 5]], 0.3)
    m = AdaptiveRadiusMC(init_radius=0.5, decay=1.0, sparse_weight=0.0, maintenance_interval=25).fit_stream(X)
    assert m.s0.sum() == pytest.approx(300.0)
    assert m.s1.sum(axis=0) == pytest.approx(X.sum(axis=0))


def test_radius_adapts_to_local_scale():
    # one tight and one broad blob: broad micro-clusters must end up with
    # larger radii than tight ones (the whole point of a local radius).
    rng = np.random.default_rng(1)
    tight = rng.normal(0, 0.05, (1500, 2))
    broad = rng.normal(0, 1.0, (1500, 2)) + np.array([20.0, 0.0])
    X = np.vstack([tight, broad])[rng.permutation(3000)]
    m = AdaptiveRadiusMC(init_radius=0.5, max_radius=3.0, decay=1.0, maintenance_interval=50).fit_stream(X)
    r, c = m._radii(), m._centers()
    heavy = m.s0 >= 20
    assert r[heavy & (c[:, 0] < 10)].max() < r[heavy & (c[:, 0] > 10)].min()


def test_memory_is_bounded_by_max_micro():
    rng = np.random.default_rng(2)
    X = rng.uniform(-50, 50, (2000, 2))  # adversarial: every point far from the rest
    m = AdaptiveRadiusMC(init_radius=0.1, max_micro=40, maintenance_interval=25).fit_stream(X)
    assert m.n_micro <= 40


def test_idle_micro_clusters_are_pruned():
    m = AdaptiveRadiusMC(init_radius=0.5, decay=0.9, sparse_weight=0.2, maintenance_interval=10)
    for _ in range(5):
        m.partial_fit(np.array([100.0, 100.0]))
    for _ in range(200):
        m.partial_fit(np.array([0.0, 0.0]))
    assert not np.any(np.linalg.norm(m._centers() - np.array([100.0, 100.0]), axis=1) < 1.0)


def test_overlapping_micro_clusters_merge():
    m = AdaptiveRadiusMC(init_radius=0.5, merge_factor=0.5, maintenance_interval=10_000, decay=1.0)
    # two weight-5 micro-clusters whose centers are 0.1 apart (< 0.5 * radius)
    m.s0 = np.array([5.0, 5.0])
    m.s1 = np.array([[0.0, 0.0], [0.1, 0.0]]) * 5.0
    m.s2 = np.array([0.0, 0.05])
    m.labels = np.array([-1, -1])
    assert m._merge_overlapping() == 1
    assert m.n_micro == 1
    assert m.s0[0] == pytest.approx(10.0)


def test_separates_two_far_blobs_and_assigns_same_blob_consistently():
    X, k = _blobs(1200, [[0, 0], [8, 8]], 0.3)
    m = AdaptiveRadiusMC(init_radius=0.5, maintenance_interval=50).fit_stream(X)
    assert m.summary()["n_clusters"] == 2
    a = m.predict_point_cluster(np.array([0.0, 0.0]))
    b = m.predict_point_cluster(np.array([8.0, 8.0]))
    assert a is not None and b is not None and a != b


def test_far_outlier_is_unassigned():
    X, _ = _blobs(600, [[0, 0]], 0.3)
    m = AdaptiveRadiusMC(init_radius=0.5, maintenance_interval=50).fit_stream(X)
    assert m.predict_point_cluster(np.array([100.0, 100.0])) is None


def test_works_in_higher_dimension_without_exponential_blowup():
    X, _ = _blobs(800, [[0.0] * 12, [6.0] * 12], 0.4)
    m = AdaptiveRadiusMC(init_radius=1.5, max_micro=200, maintenance_interval=50).fit_stream(X)
    assert m.n_micro <= 200
    assert m.summary()["n_clusters"] >= 1


def test_summary_does_not_change_predictions():
    X, _ = _blobs(400, [[0, 0], [5, 5]], 0.3)
    m = AdaptiveRadiusMC(init_radius=0.5, maintenance_interval=50).fit_stream(X[:370])
    before = m.labels.copy()
    m.summary()
    assert np.array_equal(before, m.labels)
