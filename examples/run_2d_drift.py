from pathlib import Path
import numpy as np
from adaptive_dstream import AdaptiveDStream, make_drifting_stream
from adaptive_dstream.plotting import plot_state


def main():
    X, y, phase = make_drifting_stream(n_samples=3000, random_state=42)
    model = AdaptiveDStream(
        lower=np.array([-6.0, -6.0]),
        upper=np.array([6.0, 6.0]),
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
    for t, x in enumerate(X, start=1):
        model.partial_fit(x, t=t)
        if t in checkpoints:
            recent = X[max(0, t - 400):t]
            fig, _ = plot_state(
                model,
                recent_points=recent,
                title=f"Adaptive D-Stream v0 — t={t}",
                save_path=str(output_dir / f"state_t{t}.png"),
            )
            plt = __import__("matplotlib.pyplot", fromlist=["close"])
            plt.close(fig)
            print(f"t={t}: {model.summary()}")
    print("Final summary:", model.summary())


if __name__ == "__main__":
    main()
