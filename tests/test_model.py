import numpy as np
import pytest
from adaptive_dstream import AdaptiveDStream
from adaptive_dstream.cell import GridCell


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
        merge_threshold=-2.0,
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


def test_invalid_split_strategy_raises():
    with pytest.raises(ValueError):
        AdaptiveDStream(
            lower=np.array([-1.0, -1.0]), upper=np.array([1.0, 1.0]), split_strategy="bogus",
        )


def test_merge_threshold_must_be_below_split_threshold():
    with pytest.raises(ValueError):
        AdaptiveDStream(
            lower=np.array([-1.0, -1.0]), upper=np.array([1.0, 1.0]),
            split_threshold=0.04, merge_threshold=0.04,
        )


def test_point_mass_split_puts_all_mass_in_one_child():
    model = AdaptiveDStream(
        lower=np.array([-1.0, -1.0]), upper=np.array([1.0, 1.0]),
        split_threshold=0.04, max_depth=1, max_cells=10, split_strategy="point_mass",
    )
    for t in range(1, 6):
        model.partial_fit(np.array([0.7, 0.7]), t=t)
    children = model.root.children
    assert len(children) == 4
    nonzero = [c for c in children if c.s0 > 0.0]
    assert len(nonzero) == 1
    # Mass is conserved exactly: nothing is dropped or duplicated.
    assert nonzero[0].s0 == pytest.approx(sum(c.s0 for c in children))
    # The point (0.7, 0.7) falls in the upper-right quadrant.
    assert nonzero[0].contains(np.array([0.7, 0.7]))


def test_moment_based_split_conserves_mass_and_skews_toward_the_mean():
    model = AdaptiveDStream(
        lower=np.array([-1.0, -1.0]), upper=np.array([1.0, 1.0]),
        split_threshold=0.04, max_depth=1, max_cells=10, split_strategy="moment_based",
    )
    for t in range(1, 6):
        model.partial_fit(np.array([0.7, 0.7]), t=t)
    children = model.root.children
    assert len(children) == 4
    assert sum(c.s0 for c in children) > 0.0
    # The quadrant containing the tracked mean (0.7, 0.7) gets the most mass,
    # unlike equal_uniform which would split it evenly across all four.
    upper_right = next(c for c in children if c.contains(np.array([0.7, 0.7])))
    others = [c for c in children if c is not upper_right]
    assert upper_right.s0 > max(c.s0 for c in others)


def test_moment_based_split_conserves_total_mass_exactly():
    # Direct, controlled check (bypassing partial_fit's decay/noise) that
    # the per-axis truncated-normal mass fractions sum to exactly 1 across
    # the 2**d children, so no mass is created or lost by the split.
    model = AdaptiveDStream(
        lower=np.array([-1.0, -1.0]), upper=np.array([1.0, 1.0]), split_strategy="moment_based",
    )
    cell = GridCell(np.array([-1.0, -1.0]), np.array([1.0, 1.0]), level=0, last_update=0)
    cell.s0 = 20.0
    cell.s1 = np.array([12.0, 12.0])  # mean = (0.6, 0.6)
    cell.s2 = np.array([8.0, 8.0])    # variance = 8/20 - 0.6^2 = 0.04
    cell.raw_count = 20
    parent_mass = cell.s0
    model._split(cell)
    assert len(cell.children) == 4
    assert sum(c.s0 for c in cell.children) == pytest.approx(parent_mass, rel=1e-9)


def test_contraction_merges_stale_split_back_into_leaf():
    model = AdaptiveDStream(
        lower=np.array([-1.0, -1.0]), upper=np.array([1.0, 1.0]),
        split_threshold=0.04, merge_threshold=0.03, merge_min_age=10,
        max_depth=1, max_cells=10,
    )
    # Exactly 4 points: raw_count hits the split gate (max(4, 2*dim)=4) on
    # the 4th, triggering the split with no 5th point landing in a child
    # afterward — every child's score is then exactly 0 (matches the
    # assumed-uniform baseline by construction), isolating this test to the
    # merge_threshold/merge_min_age logic rather than genuine post-split data.
    for t in range(1, 5):
        model.partial_fit(np.array([0.7, 0.7]), t=t)
    assert len(model.leaves()) == 4
    assert not model.root.is_leaf

    # Simulate the region going idle: advance the clock as if no further
    # points arrived, past merge_min_age, then run maintenance directly.
    model.t = model.t + model.merge_min_age + 1
    model.maintenance()

    assert model.root.is_leaf
    assert len(model.leaves()) == 1


def test_contraction_respects_merge_min_age():
    model = AdaptiveDStream(
        lower=np.array([-1.0, -1.0]), upper=np.array([1.0, 1.0]),
        split_threshold=0.04, merge_threshold=0.03, merge_min_age=1000,
        max_depth=1, max_cells=10,
    )
    for t in range(1, 5):
        model.partial_fit(np.array([0.7, 0.7]), t=t)
    assert not model.root.is_leaf

    model.t = model.t + 5  # well under merge_min_age
    model.maintenance()
    assert not model.root.is_leaf


def test_transitional_cell_joins_adjacent_dense_cluster():
    model = AdaptiveDStream(
        lower=np.array([0.0, 0.0]), upper=np.array([2.0, 1.0]),
        dense_threshold=5.0, sparse_threshold=1.0,
    )
    dense_cell = GridCell(np.array([0.0, 0.0]), np.array([1.0, 1.0]), level=1, last_update=0)
    dense_cell.s0 = 10.0
    transitional_cell = GridCell(np.array([1.0, 0.0]), np.array([2.0, 1.0]), level=1, last_update=0)
    transitional_cell.s0 = 3.0  # between sparse_threshold=1.0 and dense_threshold=5.0
    model.root.children = [dense_cell, transitional_cell]

    model._assign_clusters()
    assert dense_cell.cluster_id == 0
    assert transitional_cell.cluster_id == 0

    predicted = model.predict_point_cluster(np.array([1.5, 0.5]))
    assert predicted == 0


def test_transitional_cell_with_no_dense_neighbor_stays_unassigned():
    model = AdaptiveDStream(
        lower=np.array([0.0, 0.0]), upper=np.array([2.0, 1.0]),
        dense_threshold=5.0, sparse_threshold=1.0,
    )
    lonely_cell = GridCell(np.array([0.0, 0.0]), np.array([2.0, 1.0]), level=1, last_update=0)
    lonely_cell.s0 = 3.0  # transitional, but no dense neighbor exists
    model.root.children = [lonely_cell]

    model._assign_clusters()
    assert lonely_cell.cluster_id is None


def test_density_normalize_off_by_default_is_a_no_op():
    model = AdaptiveDStream(lower=np.array([-1.0, -1.0]), upper=np.array([1.0, 1.0]))
    assert model.density_normalize is False
    child = GridCell(np.array([0.0, 0.0]), np.array([1.0, 1.0]), level=1, last_update=0)
    assert model._volume_scale(child) == pytest.approx(1.0)


def test_density_normalize_scales_effective_count_by_inverse_relative_volume():
    # Splitting fragments a region's mass across 2**d children, so a cell
    # covering 1/4 of the domain (half-width per axis, in 2D) should need
    # only ~1/4 the raw count to read as equally dense once density_normalize
    # rescales counts to a shared (root) reference volume.
    model = AdaptiveDStream(
        lower=np.array([-1.0, -1.0]), upper=np.array([1.0, 1.0]),
        dense_threshold=1.0, sparse_threshold=0.1, density_normalize=True,
    )
    assert model._volume_scale(model.root) == pytest.approx(1.0)  # root == domain volume

    quarter_cell = GridCell(np.array([0.0, 0.0]), np.array([1.0, 1.0]), level=1, last_update=0)
    scale = model._volume_scale(quarter_cell)
    assert scale == pytest.approx(4.0)

    quarter_cell.s0 = 0.26  # > dense_threshold / 4, so dense once rescaled...
    assert quarter_cell.state(model.t, model.decay, model.dense_threshold,
                               model.sparse_threshold, scale) == "dense"
    # ...but the identical raw count would not clear the threshold unscaled.
    assert quarter_cell.state(model.t, model.decay, model.dense_threshold,
                               model.sparse_threshold, 1.0) == "transitional"


# merge-driven shrink, then prune-only shrink
@pytest.mark.parametrize("merge_min_age,idle_prune_after", [(20, 30), (10**9, 30)])
def test_running_leaf_counter_matches_tree_through_split_merge_prune(merge_min_age, idle_prune_after):
    rng = np.random.default_rng(0)
    model = AdaptiveDStream(
        lower=np.array([-1.0, -1.0]), upper=np.array([1.0, 1.0]),
        split_threshold=0.04, merge_threshold=0.01, merge_min_age=merge_min_age,
        decay=0.9, maintenance_interval=10, idle_prune_after=idle_prune_after, sparse_threshold=2.0, max_depth=4,
    )
    seen_splits = seen_shrinks = 0
    prev = model.n_leaves
    # Two phases in different corners so early splits go idle and get merged/pruned.
    for t in range(1, 1201):
        center = np.array([0.6, 0.6]) if t <= 600 else np.array([-0.6, -0.6])
        model.partial_fit(np.clip(rng.normal(center, 0.15), -1, 1), t=t)
        assert model.n_leaves == len(model.leaves())
        seen_splits += model.n_leaves > prev
        seen_shrinks += model.n_leaves < prev
        prev = model.n_leaves
    assert seen_splits > 0 and seen_shrinks > 0


def test_clustering_runs_only_at_checkpoints_not_per_prediction():
    rng = np.random.default_rng(1)
    model = AdaptiveDStream(
        lower=np.array([-1.0, -1.0]), upper=np.array([1.0, 1.0]),
        maintenance_interval=50, dense_threshold=3.0, sparse_threshold=0.5,
    )
    calls = []
    original = model._assign_clusters
    model._assign_clusters = lambda: (calls.append(model.t), original())[1]
    preds = []
    for t in range(1, 150):
        x = np.clip(rng.normal([0.5, 0.5], 0.05), -1, 1)
        preds.append(model.predict_point_cluster(x))
        model.partial_fit(x, t=t)
    assert calls == [50, 100]  # checkpoints only; 149 predictions triggered no clustering
    assert all(p is None for p in preds[:50])  # nothing clustered before the first checkpoint
    assert preds[-1] is not None  # dense blob is labelled once a checkpoint has run


def test_split_children_inherit_parent_cluster_id():
    model = AdaptiveDStream(lower=np.array([-1.0, -1.0]), upper=np.array([1.0, 1.0]))
    model.root.cluster_id = 7
    model._split(model.root)
    assert [c.cluster_id for c in model.root.children] == [7, 7, 7, 7]
