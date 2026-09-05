import numpy as np
import pytest

from adaptive_dstream import FixedGridDStream
from adaptive_dstream.baselines import RiverClusterAdapter


def test_fixed_grid_builds_exact_cell_count_and_never_splits():
    model = FixedGridDStream(
        lower=np.array([-1.0, -1.0]),
        upper=np.array([1.0, 1.0]),
        n_cells_per_dim=5,
    )
    assert len(model.leaves()) == 25
    for t in range(1, 200):
        model.partial_fit(np.array([0.9, 0.9]), t=t)
    # Repeatedly hammering one corner cell must not create new cells.
    assert len(model.leaves()) == 25


def test_fixed_grid_never_prunes_idle_cells():
    # A fixed grid's defining property is a partition that never changes.
    # AdaptiveDStream.maintenance() prunes idle+sparse leaves, which would
    # otherwise delete grid cells that just haven't seen a point recently —
    # breaking the "fixed resolution" contract and leaving spatial gaps.
    model = FixedGridDStream(
        lower=np.array([-1.0, -1.0]),
        upper=np.array([1.0, 1.0]),
        n_cells_per_dim=5,
        idle_prune_after=10,
        maintenance_interval=5,
    )
    assert len(model.leaves()) == 25
    # Only ever touch one cell; every other cell is idle+sparse throughout,
    # for far longer than idle_prune_after, across several maintenance calls.
    for t in range(1, 101):
        model.partial_fit(np.array([0.9, 0.9]), t=t)
    assert len(model.leaves()) == 25
    assert len(model.root.children) == 25


def test_fixed_grid_cells_tile_the_domain_without_gaps():
    model = FixedGridDStream(
        lower=np.array([0.0, 0.0]),
        upper=np.array([4.0, 4.0]),
        n_cells_per_dim=4,
    )
    total_area = sum(np.prod(c.side_lengths) for c in model.leaves())
    assert total_area == pytest.approx(16.0)


def test_fixed_grid_rejects_invalid_resolution():
    with pytest.raises(ValueError):
        FixedGridDStream(lower=np.array([0.0]), upper=np.array([1.0]), n_cells_per_dim=0)


def test_river_adapter_predicts_and_reports_cluster_count():
    river = pytest.importorskip("river")
    from river import cluster

    adapter = RiverClusterAdapter(cluster.DBSTREAM(), name="DBSTREAM")
    rng = np.random.default_rng(0)
    for i in range(100):
        x = rng.normal(size=2)
        adapter.partial_fit(x, t=i + 1)
    pred = adapter.predict_point_cluster(np.array([0.0, 0.0]))
    assert pred is None or isinstance(pred, int)
    summary = adapter.summary()
    assert "n_leaves" in summary
