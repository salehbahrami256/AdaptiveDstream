import numpy as np
from adaptive_dstream import AdaptiveDStream


def test_basic_updates():
    model = AdaptiveDStream(
        lower=np.array([-2.0, -2.0]),
        upper=np.array([2.0, 2.0]),
        split_threshold=999.0,
    )
    model.partial_fit(np.array([0.1, 0.2]), t=1)
    model.partial_fit(np.array([0.2, 0.3]), t=2)
    assert model.n_seen == 2
    assert len(model.leaves()) == 1
    assert model.root.s0 > 1.0


def test_split_creates_four_children_in_2d():
    model = AdaptiveDStream(
        lower=np.array([-1.0, -1.0]),
        upper=np.array([1.0, 1.0]),
        split_threshold=-1.0,
        max_depth=1,
        max_cells=10,
    )
    for t in range(1, 6):
        model.partial_fit(np.array([0.7, 0.7]), t=t)
    assert len(model.leaves()) == 4


def test_out_of_domain_point_is_clipped_not_rejected():
    # Unbounded distributions (Gaussian tails) will occasionally land
    # outside any finite domain box. partial_fit/predict must clip to the
    # boundary rather than raising, since crashing on a rare tail point
    # would make the model unusable for real streaming evaluation.
    model = AdaptiveDStream(
        lower=np.array([-1.0, -1.0]),
        upper=np.array([1.0, 1.0]),
        split_threshold=999.0,
    )
    far_point = np.array([50.0, -50.0])
    model.partial_fit(far_point, t=1)
    assert model.n_seen == 1
    assert model.root.s0 > 0.0
    # predict_point_cluster must not raise either.
    model.predict_point_cluster(far_point)
