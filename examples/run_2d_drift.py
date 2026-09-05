import argparse
from pathlib import Path
import numpy as np
from adaptive_dstream import AdaptiveDStream, make_drifting_stream
from adaptive_dstream.plotting import plot_state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dim", type=int, default=2,
                         help="Space dimension (default: 2 — kept at 2 for the README, since "
                              "the grid partition itself is only drawable up to 2D; see "
                              "plotting.plot_state for what happens at higher dimensions).")
    args = parser.parse_args()
    dim = args.dim

    X, y, phase = make_drifting_stream(n_samples=3000, random_state=42, dim=dim)
    model = AdaptiveDStream(
        lower=np.full(dim, -6.0),
        upper=np.full(dim, 6.0),
        decay=0.995,
        dense_threshold=1.5,
        sparse_threshold=0.2,
        split_threshold=0.04,
        max_depth=6,
        max_cells=2500,
        maintenance_interval=50,
        idle_prune_after=300,
    )
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    checkpoints = {1000, 2000, 3000}
    # Keep the original filenames for dim=2 so the README's embedded images
    # keep working; disambiguate other dimensions instead of overwriting them.
    suffix = "" if dim == 2 else f"_dim{dim}"
    for t, x in enumerate(X, start=1):
        model.partial_fit(x, t=t)
        if t in checkpoints:
            recent = X[max(0, t - 400):t]
            fig, _ = plot_state(
                model,
                recent_points=recent,
                title=f"Adaptive D-Stream v0 — t={t}",
                save_path=str(output_dir / f"state{suffix}_t{t}.png"),
            )
            plt = __import__("matplotlib.pyplot", fromlist=["close"])
            plt.close(fig)
            print(f"t={t}: {model.summary()}")
    print("Final summary:", model.summary())


if __name__ == "__main__":
    main()
