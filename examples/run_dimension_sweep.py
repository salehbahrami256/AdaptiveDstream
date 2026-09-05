"""Dimension sweep: does AdaptiveDStream's memory/accuracy standing versus a
fixed grid improve as dimension increases?

A fixed-resolution grid's cell count is ``n_cells_per_dim ** dim`` —
exponential in dimension, paid unconditionally regardless of where the data
actually is. AdaptiveDStream only refines cells the data needs and is
hard-capped at ``max_cells``, so its cell count is not tied to that
exponential. To make this a fair fight rather than a straw man, the fixed
grid's resolution is chosen *per dimension* so its total cell count lands
near ``CELL_BUDGET`` (the same order as AdaptiveDStream's ``max_cells``) at
every dimension — i.e. both models get roughly the same memory budget, only
the fixed grid's resolution-per-axis has to shrink to stay within it as
dimension grows. The hypothesis: at a fixed, shared memory budget, the
fixed grid's necessarily-coarser per-axis resolution should hurt it more
than AdaptiveDStream as dimension increases.

This uses the same adversarial ``varying_density`` stream as the headline
result, generalized to arbitrary dimension: the two cluster centers still
orbit in the first two coordinates (see ``synthetic._orbit_centers``), so
every dimension beyond the first two is pure isotropic noise, uninformative
for clustering — deliberately, since that is the standard curse-of-
dimensionality stress case, not an easier or harder version of the problem.

Both models use the *same* hyperparameters at every dimension (no
per-dimension retuning beyond the fixed grid's resolution described above),
so this is "same code, different dimension," not a best case for either.

Run: python examples/run_dimension_sweep.py
Outputs: outputs/dimension_sweep_results.json,
outputs/dimension_sweep_memory.png, outputs/dimension_sweep_ari.png
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

from adaptive_dstream import (
    AdaptiveDStream,
    FixedGridDStream,
    make_varying_density_stream,
    run_stream_eval,
)

SEED = 7
N_SAMPLES = 2500
DIMS = [2, 3, 4, 5, 6, 8, 10]
CELL_BUDGET = 2500  # target total fixed-grid cells; matches AdaptiveDStream's max_cells below
ADAPTIVE_MAX_CELLS = 2500

OUTPUT_DIR = Path("outputs")


def fixed_resolution_for_budget(dim: int, budget: int) -> int:
    """n_cells_per_dim such that n**dim is close to (and at least 2**dim)."""
    return max(2, round(budget ** (1.0 / dim)))


def main() -> None:
    results = []
    for dim in DIMS:
        gen_params = dict(dense_std=0.18, sparse_std=1.6, dense_weight=0.5, n_phases=3, dim=dim)
        X, y, phase = make_varying_density_stream(n_samples=N_SAMPLES, random_state=SEED, **gen_params)
        margin = 1.0
        lower = X.min(axis=0) - margin
        upper = X.max(axis=0) + margin

        common = dict(lower=lower, upper=upper, decay=0.99, maintenance_interval=50, idle_prune_after=400)

        n_cells = fixed_resolution_for_budget(dim, CELL_BUDGET)
        total_cells = n_cells ** dim
        factory = lambda n=n_cells, kw=common: FixedGridDStream(
            n_cells_per_dim=n, dense_threshold=2.0, sparse_threshold=0.3, **kw)
        r = run_stream_eval(factory, X, y, phase=phase, name="FixedGrid(budget-matched)")
        results.append({**r.to_dict(), "family": "FixedGridDStream", "dim": dim,
                         "n_cells_per_dim": n_cells, "total_cells": total_cells})
        print(f"dim={dim:>2}  FixedGrid n={n_cells} ({total_cells} cells)  ARI={r.ari:.3f}  "
              f"peak_mem={r.peak_memory_bytes/1024:.1f}KB  unassigned={r.fraction_unassigned:.2f}")

        adaptive_factory = lambda kw=common: AdaptiveDStream(
            dense_threshold=0.5, sparse_threshold=0.05, split_threshold=0.05,
            max_depth=7, max_cells=ADAPTIVE_MAX_CELLS, **kw)
        r = run_stream_eval(adaptive_factory, X, y, phase=phase, name="AdaptiveDStream")
        results.append({**r.to_dict(), "family": "AdaptiveDStream", "dim": dim,
                         "n_cells_per_dim": None, "total_cells": r.active_cells_over_time[-1][1]})
        print(f"dim={dim:>2}  AdaptiveDStream               ARI={r.ari:.3f}  "
              f"peak_mem={r.peak_memory_bytes/1024:.1f}KB  unassigned={r.fraction_unassigned:.2f}")

    OUTPUT_DIR.mkdir(exist_ok=True)
    with open(OUTPUT_DIR / "dimension_sweep_results.json", "w") as f:
        json.dump(results, f, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else o)

    series = [("AdaptiveDStream", "tab:red", "*", 12), ("FixedGrid(budget-matched)", "tab:blue", "o", 7)]

    for metric, ylabel, fname, log in [
        ("peak_memory_bytes", "Peak memory (bytes)", "dimension_sweep_memory.png", True),
        ("ari", "Adjusted Rand Index", "dimension_sweep_ari.png", False),
    ]:
        fig, ax = plt.subplots(figsize=(7, 5))
        for name, color, marker, size in series:
            rows = sorted((r for r in results if r["name"] == name), key=lambda r: r["dim"])
            if not rows:
                continue
            ax.plot([r["dim"] for r in rows], [r[metric] for r in rows],
                    marker=marker, color=color, label=name, markersize=size)
        if log:
            ax.set_yscale("log")
        ax.set_xlabel("Dimension")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel} vs. dimension\nvarying-density stream, ~{CELL_BUDGET}-cell budget at every dimension")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / fname, dpi=150)
        plt.close(fig)

    print(f"\nWrote {OUTPUT_DIR}/dimension_sweep_results.json and dimension_sweep_{{memory,ari}}.png")


if __name__ == "__main__":
    main()
