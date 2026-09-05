import numpy as np

from adaptive_dstream.synthetic import (
    generate_stream,
    load_stream,
    make_moons_stream,
    make_varying_density_stream,
    save_stream,
)


def test_varying_density_stream_is_reproducible():
    X1, y1, phase1 = make_varying_density_stream(n_samples=500, random_state=123)
    X2, y2, phase2 = make_varying_density_stream(n_samples=500, random_state=123)
    np.testing.assert_array_equal(X1, X2)
    np.testing.assert_array_equal(y1, y2)
    np.testing.assert_array_equal(phase1, phase2)


def test_varying_density_stream_has_both_labels_and_dense_cluster_is_tighter():
    X, y, phase = make_varying_density_stream(n_samples=2000, random_state=1, dense_std=0.15, sparse_std=1.5)
    assert set(np.unique(y)) == {0, 1}
    dense_spread = X[y == 0].std(axis=0).mean()
    sparse_spread = X[y == 1].std(axis=0).mean()
    assert dense_spread < sparse_spread


def test_moons_stream_shapes():
    X, y, phase = make_moons_stream(n_samples=300, random_state=0, n_phases=3)
    assert X.shape == (300, 2)
    assert set(np.unique(y)) == {0, 1}
    assert set(np.unique(phase)) == {0, 1, 2}


def test_generate_stream_dispatch_matches_direct_call():
    X1, y1, phase1 = generate_stream("varying_density", n_samples=200, random_state=5)
    X2, y2, phase2 = make_varying_density_stream(n_samples=200, random_state=5)
    np.testing.assert_array_equal(X1, X2)
    np.testing.assert_array_equal(y1, y2)
    np.testing.assert_array_equal(phase1, phase2)


def test_save_and_load_stream_roundtrip(tmp_path):
    X, y, phase = make_varying_density_stream(n_samples=150, random_state=9)
    save_stream(tmp_path, "stream_a", X, y, phase, generator="varying_density", seed=9, params={"dense_std": 0.18})

    X_loaded, y_loaded, phase_loaded, manifest = load_stream(tmp_path, "stream_a")
    np.testing.assert_array_equal(X, X_loaded)
    np.testing.assert_array_equal(y, y_loaded)
    np.testing.assert_array_equal(phase, phase_loaded)
    assert manifest["seed"] == 9
    assert manifest["generator"] == "varying_density"
    assert manifest["n_samples"] == 150

    # The manifest alone must be enough to regenerate the identical stream.
    X_regen, y_regen, phase_regen = generate_stream(
        manifest["generator"], n_samples=manifest["n_samples"], random_state=manifest["seed"], **manifest["params"]
    )
    np.testing.assert_array_equal(X, X_regen)
    np.testing.assert_array_equal(y, y_regen)
    np.testing.assert_array_equal(phase, phase_regen)
