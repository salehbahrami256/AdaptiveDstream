from .model import AdaptiveDStream
from .baselines import FixedGridDStream, RiverClusterAdapter, make_river_baselines
from .synthetic import (
    make_drifting_stream,
    make_varying_density_stream,
    make_moons_stream,
    generate_stream,
    save_stream,
    load_stream,
)
from .evaluation import run_stream_eval, purity_score, EvalResult
from .criteria import (
    moment_refinement_score,
    weyl_equidistribution_score,
    weyl_threshold,
    WeylTestResult,
    random_projection_score,
    RandomProjectionTestResult,
)
from .cell_shapes import sample_shape, SHAPES
from .real_data import load as load_real_dataset, LOADERS as REAL_DATASETS

__all__ = [
    "AdaptiveDStream",
    "FixedGridDStream",
    "RiverClusterAdapter",
    "make_river_baselines",
    "make_drifting_stream",
    "make_varying_density_stream",
    "make_moons_stream",
    "generate_stream",
    "save_stream",
    "load_stream",
    "run_stream_eval",
    "purity_score",
    "EvalResult",
    "moment_refinement_score",
    "weyl_equidistribution_score",
    "weyl_threshold",
    "WeylTestResult",
    "random_projection_score",
    "RandomProjectionTestResult",
    "sample_shape",
    "SHAPES",
    "load_real_dataset",
    "REAL_DATASETS",
]
