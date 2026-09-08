"""Cross-regime comparison: same models, same hyperparameters, five different
data-generating regimes.

The frontier sweep (`run_frontier_sweep.py`) asks "how does accuracy trade
off against memory on one adversarial stream." This script asks a different
question: "how much does a fixed hyperparameter choice generalize across
qualitatively different regimes" -- static vs. drifting, spatial drift vs.
density-only drift, and an adversarial density mismatch vs. a symmetric one.
No model's hyperparameters are retuned per regime; every regime uses the same
`AdaptiveDStream` config (the one selected in `run_frontier_sweep.py`) and the
same two `FixedGridDStream` resolutions, so what varies is only the data.

Regimes:
  static_no_drift    -- varying-density stream (tight+broad cluster,
                         simultaneous), `drift=False`: both cluster centers
                         frozen for the whole run. The no-drift control.
  spatial_drift       -- the headline varying-density stream: same tight and
                         broad clusters, centers now drifting across phases
                         (this is `run_frontier_sweep.py`'s stream, at a
                         smaller sample count for this sweep's runtime).
  density_drift        -- varying-density stream with centers frozen
                         (`drift=False`) but the dense cluster's std widening
                         phase-by-phase via `dense_std_schedule`: density
                         changes without any relocation.
  two_gaussian_drift   -- the original symmetric two-Gaussian stream
                         (`make_drifting_stream`): both clusters have equal,
                         constant spread. Not adversarial for a fixed grid --
                         included as a control for how much of AdaptiveDStream's
                         standing in the other regimes is about density
                         *mismatch* specifically, not drift in general.
  moons_drift          -- two rotating, drifting non-convex "moon" clusters
                         (`make_moons_stream`): arbitrary shape rather than
                         Gaussian blobs.

Run: python examples/run_regime_sweep.py
Outputs: outputs/regime_sweep_results.json, outputs/regime_sweep_ari.png
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from adaptive_dstream import (
    AdaptiveDStream,
    FixedGridDStream,
    make_river_baselines,
    run_stream_eval,
)
from adaptive_dstream.synthetic import (
    make_drifting_stream,
    make_moons_stream,
    make_varying_density_stream,
)

SEED = 7
# Reduced from 2000: see run_frontier_sweep.py's N_SAMPLES comment -- with
# contraction and transitional-cell assignment now active, this is 5
# regimes x 6 models, each paying the same per-point cost that made a single
# 4000-point AdaptiveDStream run take well over a minute.
N_SAMPLES = 500
FIXED_GRID_RESOLUTIONS = [8, 32]
OUTPUT_DIR = Path("outputs")


def build_regimes(seed: int):
    regimes = {}

    regimes["static_no_drift"] = make_varying_density_stream(
        n_samples=N_SAMPLES, random_state=seed,
        dense_std=0.18, sparse_std=1.6, dense_weight=0.5, n_phases=3, drift=False,
    )
    regimes["spatial_drift"] = make_varying_density_stream(
        n_samples=N_SAMPLES, random_state=seed,
        dense_std=0.18, sparse_std=1.6, dense_weight=0.5, n_phases=3, drift=True,
    )
    regimes["density_drift"] = make_varying_density_stream(
        n_samples=N_SAMPLES, random_state=seed,
        dense_weight=0.5, n_phases=3, drift=False,
        dense_std_schedule=[0.15, 0.15, 1.1], sparse_std_schedule=[1.6, 1.6, 1.6],
    )
    regimes["two_gaussian_drift"] = make_drifting_stream(n_samples=N_SAMPLES, random_state=seed)
    regimes["moons_drift"] = make_moons_stream(n_samples=N_SAMPLES, random_state=seed, n_phases=3)
    return regimes


def main() -> None:
    regimes = build_regimes(SEED)
    results = []

    for regime_name, (X, y, phase) in regimes.items():
        margin = 1.0
        lower = X.min(axis=0) - margin
        upper = X.max(axis=0) + margin

        common = dict(
            lower=lower, upper=upper, decay=0.99,
            maintenance_interval=50, idle_prune_after=400,
        )

        print(f"\n=== regime: {regime_name} (n={N_SAMPLES}) ===")

        for n_cells in FIXED_GRID_RESOLUTIONS:
            factory = lambda n=n_cells: FixedGridDStream(
                n_cells_per_dim=n, dense_threshold=2.0, sparse_threshold=0.3, **common,
            )
            r = run_stream_eval(factory, X, y, phase=phase, name=f"FixedGrid(n={n_cells})")
            results.append({**r.to_dict(), "regime": regime_name, "family": "FixedGridDStream",
                             "n_cells_per_dim": n_cells})
            print(f"  FixedGrid n={n_cells:>3}  ARI={r.ari:.3f}  NMI={r.nmi:.3f}  "
                  f"peak_mem={r.peak_memory_bytes/1024:.1f}KB  unassigned={r.fraction_unassigned:.2f}")

        # Same AdaptiveDStream hyperparameters in every regime -- see module
        # docstring. Values match the tuned config in run_frontier_sweep.py.
        adaptive_factory = lambda: AdaptiveDStream(
            **common, dense_threshold=0.5, sparse_threshold=0.05,
            split_threshold=0.05, max_depth=7, max_cells=3000,
            merge_threshold=0.01, merge_min_age=200,
        )
        r = run_stream_eval(adaptive_factory, X, y, phase=phase, name="AdaptiveDStream")
        results.append({**r.to_dict(), "regime": regime_name, "family": "AdaptiveDStream",
                         "n_cells_per_dim": None})
        print(f"  AdaptiveDStream      ARI={r.ari:.3f}  NMI={r.nmi:.3f}  "
              f"peak_mem={r.peak_memory_bytes/1024:.1f}KB  unassigned={r.fraction_unassigned:.2f}")

        for name, adapter in make_river_baselines(seed=SEED).items():
            factory = (lambda a=adapter: a)
            r = run_stream_eval(factory, X, y, phase=phase, name=name)
            results.append({**r.to_dict(), "regime": regime_name, "family": name, "n_cells_per_dim": None})
            print(f"  {name:<12}         ARI={r.ari:.3f}  NMI={r.nmi:.3f}  "
                  f"peak_mem={r.peak_memory_bytes/1024:.1f}KB  unassigned={r.fraction_unassigned:.2f}")

    OUTPUT_DIR.mkdir(exist_ok=True)
    with open(OUTPUT_DIR / "regime_sweep_results.json", "w") as f:
        json.dump(results, f, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else o)

    regime_order = list(regimes.keys())
    families = ["FixedGrid(n=8)", "FixedGrid(n=32)", "AdaptiveDStream", "DenStream", "CluStream", "DBSTREAM"]

    def row_key(r):
        return r["name"] if r["family"] not in ("FixedGridDStream",) else f"FixedGrid(n={r['n_cells_per_dim']})"

    fig, ax = plt.subplots(figsize=(11, 6))
    width = 0.13
    x = np.arange(len(regime_order))
    colors = ["tab:blue", "tab:cyan", "tab:red", "tab:green", "tab:orange", "tab:purple"]
    for i, fam in enumerate(families):
        vals = []
        for rn in regime_order:
            match = [r for r in results if r["regime"] == rn and row_key(r) == fam]
            vals.append(match[0]["ari"] if match else 0.0)
        ax.bar(x + (i - len(families) / 2) * width, vals, width, label=fam, color=colors[i])

    ax.set_xticks(x)
    ax.set_xticklabels(regime_order, rotation=15, ha="right")
    ax.set_ylabel("Adjusted Rand Index")
    ax.set_title("ARI by model across data-generating regimes")
    ax.legend(fontsize=8, ncol=3)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "regime_sweep_ari.png", dpi=150)
    plt.close(fig)

    print(f"\nWrote {OUTPUT_DIR}/regime_sweep_results.json and regime_sweep_ari.png")


if __name__ == "__main__":
    main()
