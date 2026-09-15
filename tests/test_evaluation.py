import numpy as np
import pytest

from adaptive_dstream import FixedGridDStream, run_stream_eval
from adaptive_dstream.evaluation import UNASSIGNED_LABEL, decayed_purity_score, purity_score


def test_purity_score_toy_example():
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_pred = np.array([0, 0, 1, 1, 1, 1])
    # cluster 0 -> {0,0} majority 2; cluster 1 -> {0,1,1,1} majority 3
    assert purity_score(y_true, y_pred) == pytest.approx(5 / 6)


def test_purity_score_perfect_partition_is_one():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([5, 5, 9, 9])
    assert purity_score(y_true, y_pred) == pytest.approx(1.0)


def test_decayed_purity_score_matches_flat_purity_when_decay_is_one():
    # decay=1.0 -> every point weighted equally -> exactly the flat metric.
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 3, size=40)
    y_pred = rng.integers(0, 3, size=40)
    assert decayed_purity_score(y_true, y_pred, decay=1.0) == pytest.approx(purity_score(y_true, y_pred))


def test_decayed_purity_score_weights_recent_points_more():
    # First half of the stream is entirely wrong, second half entirely
    # right. A flat metric sees 50% correct; a steep decay should score
    # this close to 1.0 since only the (correct) tail carries any weight.
    y_true = np.array([0] * 20 + [1] * 20)
    y_pred = np.array([1] * 20 + [1] * 20)  # first half mislabeled, second half correct
    flat = purity_score(y_true, y_pred)
    decayed = decayed_purity_score(y_true, y_pred, decay=0.5)
    assert flat == pytest.approx(0.5)
    assert decayed > 0.9


def test_decayed_purity_score_empty_is_nan():
    assert np.isnan(decayed_purity_score(np.array([]), np.array([]), decay=0.9))


class _ConstantModel:
    """Always assigns everything to cluster 0; used to test the harness in
    isolation from any real clustering logic."""

    def __init__(self):
        self.summary_calls = 0

    def predict_point_cluster(self, x):
        return 0

    def partial_fit(self, x, t=None):
        return self

    def summary(self):
        self.summary_calls += 1
        return {"n_leaves": 1}


class _AlwaysUnassignedModel:
    def predict_point_cluster(self, x):
        return None

    def partial_fit(self, x, t=None):
        return self

    def summary(self):
        return {"n_leaves": 0}


def test_run_stream_eval_perfect_single_cluster_predictor():
    X = np.zeros((50, 2))
    y = np.zeros(50, dtype=int)
    result = run_stream_eval(_ConstantModel, X, y, name="constant")
    assert result.fraction_unassigned == 0.0
    assert result.n_points == 50
    assert result.peak_memory_bytes >= 0
    assert result.throughput_pts_per_sec > 0


def test_run_stream_eval_maps_none_to_unassigned_sentinel():
    X = np.zeros((30, 2))
    y = np.array([0, 1] * 15)
    result = run_stream_eval(_AlwaysUnassignedModel, X, y, name="unassigned")
    assert np.all(result.y_pred == UNASSIGNED_LABEL)
    assert result.fraction_unassigned == 1.0


def test_run_stream_eval_eval_decay_defaults_to_none():
    X = np.zeros((20, 2))
    y = np.zeros(20, dtype=int)
    result = run_stream_eval(_ConstantModel, X, y, name="constant")
    assert result.eval_decay is None
    assert result.decayed_purity is None
    assert result.ari_recent is None
    assert result.nmi_recent is None
    assert result.recent_window is None
    d = result.to_dict()
    assert d["ari_recent"] is None and d["decayed_purity"] is None


def test_run_stream_eval_eval_decay_populates_recency_weighted_metrics():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(120, 2))
    y = (rng.random(120) > 0.5).astype(int)

    factory = lambda: FixedGridDStream(
        lower=np.array([-4.0, -4.0]), upper=np.array([4.0, 4.0]), n_cells_per_dim=4,
        dense_threshold=1.0, sparse_threshold=0.1, maintenance_interval=20,
    )
    result = run_stream_eval(factory, X, y, name="fixedgrid", eval_decay=0.9)
    assert result.eval_decay == 0.9
    # effective sample size of decay=0.9 is ceil(1/(1-0.9)) ~= 10 (11 with
    # floating-point rounding of 0.9 -- assert the formula, not a literal).
    assert result.recent_window == int(np.ceil(1.0 / (1.0 - 0.9)))
    assert -1.0 <= result.ari_recent <= 1.0
    assert 0.0 <= result.nmi_recent <= 1.0
    assert 0.0 <= result.decayed_purity <= 1.0
    d = result.to_dict()
    assert d["recent_window"] == result.recent_window


def test_run_stream_eval_tracks_active_cells_and_phase_breakdown():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(120, 2))
    y = (rng.random(120) > 0.5).astype(int)
    phase = np.array([0] * 60 + [1] * 60)

    factory = lambda: FixedGridDStream(
        lower=np.array([-4.0, -4.0]), upper=np.array([4.0, 4.0]), n_cells_per_dim=4,
        dense_threshold=1.0, sparse_threshold=0.1, maintenance_interval=20,
    )
    result = run_stream_eval(factory, X, y, phase=phase, name="fixedgrid", snapshot_every=30)
    assert len(result.active_cells_over_time) > 0
    assert all(count == 16 for _, count in result.active_cells_over_time)
    assert set(result.ari_by_phase.keys()) <= {0, 1}
    d = result.to_dict()
    assert d["name"] == "fixedgrid"
    assert -1.0 <= d["ari"] <= 1.0
    assert 0.0 <= d["nmi"] <= 1.0
