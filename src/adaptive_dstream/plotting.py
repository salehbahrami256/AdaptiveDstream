from __future__ import annotations
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def plot_state(model, recent_points=None, title=None, save_path=None):
    if model.dim != 2:
        raise ValueError("plot_state supports 2D only.")
    model._assign_clusters()
    fig, ax = plt.subplots(figsize=(8, 8))
    if recent_points is not None and len(recent_points):
        ax.scatter(recent_points[:, 0], recent_points[:, 1], s=8, alpha=0.35)
    for cell in model.leaves():
        state = cell.state(model.t, model.decay, model.dense_threshold, model.sparse_threshold)
        rect = Rectangle(
            (cell.lower[0], cell.lower[1]),
            cell.upper[0] - cell.lower[0],
            cell.upper[1] - cell.lower[1],
            fill=False,
            linewidth=1.7 if state == "dense" else 0.7,
            alpha=0.8 if state == "dense" else 0.35,
        )
        ax.add_patch(rect)
        if cell.cluster_id is not None:
            cx, cy = cell.center
            ax.text(cx, cy, str(cell.cluster_id), ha="center", va="center", fontsize=8)
    ax.set_xlim(model.lower[0], model.upper[0])
    ax.set_ylim(model.lower[1], model.upper[1])
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_title(title or f"Adaptive D-Stream at t={model.t}")
    ax.set_aspect("equal", adjustable="box")
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig, ax
